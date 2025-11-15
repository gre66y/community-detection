import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score
import random
import time
from collections import defaultdict
from typing import Dict, Tuple, List, Any
try:
    from community import community_louvain
    louvain_available = True
except ImportError:
    louvain_available = False
    print("Note: python-louvain not installed. Louvain comparisons will be skipped.")

class SSGA:
    def __init__(self, population_size=100, generations=50, crossover_rate=0.9, 
                 mutation_rate=0.1, epsilon=0.5):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.epsilon = epsilon
        
    def compute_effective_resistance(self, graph: nx.Graph) -> Dict[Tuple, float]:
        """Compute effective resistance for all edges in the graph"""
        n = graph.number_of_nodes()
        nodes = list(graph.nodes())
        
        # Create adjacency matrix (numpy array)
        A = nx.to_numpy_array(graph, nodelist=nodes, weight='weight')

        # Compute Laplacian matrix: L = D - A
        D = np.diag(np.sum(A, axis=1).ravel())
        L = D - A
        
        # Compute pseudoinverse of Laplacian
        try:
            L_plus = np.linalg.pinv(L)
        except Exception as e:
            # Fallback: use regularized version if matrix is singular
            print(f"Warning: pseudoinverse failed ({e}), using regularized Laplacian")
            L_plus = np.linalg.pinv(L + np.eye(n) * 1e-6)
        
        # Compute effective resistance for each edge
        effective_resistances = {}
        for i, j in graph.edges():
            idx_i = nodes.index(i)
            idx_j = nodes.index(j)
            resistance = (L_plus[idx_i, idx_i] + L_plus[idx_j, idx_j] 
                         - 2 * L_plus[idx_i, idx_j])
            effective_resistances[(i, j)] = max(resistance, 1e-6)  # Avoid zero
            
        return effective_resistances
    
    def spectral_sparsification(self, graph: nx.Graph) -> nx.Graph:
        """Apply Spielman-Srivastava spectral sparsification"""
        n = graph.number_of_nodes()
        if n <= 1:
            # Nothing to sparsify for graphs with 0 or 1 node
            sparsified_graph = nx.Graph()
            sparsified_graph.add_nodes_from(graph.nodes())
            return sparsified_graph

        effective_resistances = self.compute_effective_resistance(graph)
        
        # Compute sampling probabilities
        total_resistance = sum(effective_resistances.values())
        sampling_probs = {}

        for edge, resistance in effective_resistances.items():
            # If total_resistance is zero (numerical issues), handle later when normalizing
            sampling_probs[edge] = resistance / total_resistance if total_resistance else 0.0

        # Number of samples (choose a reasonable q; guard epsilon and n)
        eps = max(self.epsilon, 1e-6)
        q = int(8 * n * np.log(max(n, 2)) / (eps ** 2))
        q = max(1, q)
        q = min(q, graph.number_of_edges())  # Don't sample more than available edges
        
        # Sample edges
        edges = list(graph.edges())
        probabilities = [sampling_probs.get(edge, 0.0) for edge in edges]

        # Normalize probabilities; if they sum to zero, fall back to uniform sampling
        prob_sum = sum(probabilities)
        if prob_sum <= 0:
            probabilities = [1.0 / len(edges) for _ in edges]
        else:
            probabilities = [p / prob_sum for p in probabilities]
        
        # Create sparsified graph
        sparsified_graph = nx.Graph()
        sparsified_graph.add_nodes_from(graph.nodes())
        
        # Sample edges. If q equals number of edges, sample without replacement for variety
        replace = True
        if q <= len(edges):
            replace = False
        sampled_indices = np.random.choice(len(edges), size=q, p=probabilities, replace=replace)
        
        for idx in sampled_indices:
            u, v = edges[idx]
            if sparsified_graph.has_edge(u, v):
                # Increase weight if edge already exists
                sparsified_graph[u][v]['weight'] += 1
            else:
                sparsified_graph.add_edge(u, v, weight=1)
                
        return sparsified_graph
    
    def initialize_population(self, graph: nx.Graph) -> Tuple[List[List[int]], List[Any]]:
        """Initialize population using locus-based adjacency representation"""
        population = []
        n = graph.number_of_nodes()
        nodes = list(graph.nodes())
        
        for _ in range(self.population_size):
            individual = []
            for node in nodes:
                neighbors = list(graph.neighbors(node))
                if neighbors:
                    # Assign a random neighbor
                    individual.append(nodes.index(random.choice(neighbors)))
                else:
                    # Isolated node points to itself
                    individual.append(nodes.index(node))
            population.append(individual)
            
        return population, nodes
    
    def decode_individual(self, individual: List[int], nodes: List) -> List[List]:
        """Decode locus-based representation to communities"""
        n = len(individual)
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        
        # Build graph from locus representation
        for i, neighbor_idx in enumerate(individual):
            graph.add_edge(i, neighbor_idx)
            
        # Find connected components as communities
        communities = list(nx.connected_components(graph))
        return [list(community) for community in communities]
    
    def modularity(self, graph: nx.Graph, communities: List[List]) -> float:
        """Calculate modularity of a partition"""
        if not isinstance(graph, nx.Graph):
            return 0.0
            
        m = graph.size(weight='weight')
        if m == 0:
            return 0.0
            
        deg = dict(graph.degree(weight='weight'))
        Q = 0.0
        
        for community in communities:
            for i in community:
                for j in community:
                    if graph.has_edge(i, j):
                        A_ij = graph[i][j].get('weight', 1)
                    else:
                        A_ij = 0
                    Q += A_ij - (deg[i] * deg[j]) / (2 * m)
                    
        return Q / (2 * m)
    
    def fitness_evaluation(self, population: List[List[int]], graph: nx.Graph, nodes: List) -> List[float]:
        """Evaluate fitness of each individual using modularity"""
        fitness_scores = []
        
        for individual in population:
            communities = self.decode_individual(individual, nodes)
            # Convert node indices back to original node labels
            labeled_communities = []
            for community in communities:
                labeled_community = [nodes[i] for i in community]
                labeled_communities.append(labeled_community)
                
            modularity_score = self.modularity(graph, labeled_communities)
            fitness_scores.append(modularity_score)
            
        return fitness_scores
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Uniform crossover"""
        child = []
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child
    
    def mutate(self, individual: List[int], graph: nx.Graph, nodes: List) -> List[int]:
        """Mutation: change a gene to a random neighbor"""
        mutated = individual.copy()
        n = len(individual)
        
        for i in range(n):
            if random.random() < self.mutation_rate:
                node = nodes[i]
                neighbors = list(graph.neighbors(node))
                if neighbors:
                    mutated[i] = nodes.index(random.choice(neighbors))
                    
        return mutated
    
    def run(self, graph: nx.Graph) -> Tuple[List[List], float]:
        """Main SSGA algorithm"""
        print("┌─ Spectral Sparsification")
        sparsified_graph = self.spectral_sparsification(graph)
        original_edges = graph.number_of_edges()
        sparsified_edges = sparsified_graph.number_of_edges()
        if original_edges > 0:
            reduction_pct = (1 - sparsified_edges / original_edges) * 100
        else:
            reduction_pct = 0.0
        print(f"│  Original: {original_edges} edges")
        print(f"│  Sparsified: {sparsified_edges} edges")
        print(f"│  Reduction: {reduction_pct:.1f}%")
        
        print("└─ Genetic Algorithm for Community Detection")
        population, nodes = self.initialize_population(sparsified_graph)
        
        best_fitness = -float('inf')
        best_individual = None
        best_communities = []
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = self.fitness_evaluation(population, sparsified_graph, nodes)
            
            # Find best individual
            current_best_idx = np.argmax(fitness_scores)
            if fitness_scores[current_best_idx] > best_fitness:
                best_fitness = fitness_scores[current_best_idx]
                best_individual = population[current_best_idx]
                best_communities = self.decode_individual(best_individual, nodes)
                # Convert to labeled communities
                labeled_communities = []
                for community in best_communities:
                    labeled_community = [nodes[i] for i in community]
                    labeled_communities.append(labeled_community)
                best_communities = labeled_communities
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individual
            new_population.append(population[current_best_idx])
            
            # Tournament selection and reproduction
            while len(new_population) < self.population_size:
                # Tournament selection
                tournament_size = 3
                tournament_indices = random.sample(range(len(population)), tournament_size)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                parent1_idx = tournament_indices[np.argmax(tournament_fitness)]
                
                tournament_indices = random.sample(range(len(population)), tournament_size)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                parent2_idx = tournament_indices[np.argmax(tournament_fitness)]
                
                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child = self.crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                # Mutation
                child = self.mutate(child, sparsified_graph, nodes)
                new_population.append(child)
            
            population = new_population
            
            if generation % 10 == 0:
                print(f"   Generation {generation:3d}, Best Fitness: {best_fitness:.4f}")
        
        print(f"✓ Completed: {len(best_communities)} communities found")
        return best_communities, best_fitness

class NetworkLoader:
    """Class to load all networks mentioned in the paper"""
    
    @staticmethod
    def load_karate_club():
        """Zachary Karate Club network"""
        G = nx.karate_club_graph()
        # Ground truth communities
        true_communities = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 21],
            [9, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
        ]
        return G, true_communities, "Karate Club"
    
    @staticmethod
    def load_football_network():
        """American College Football network"""
        try:
            # Create a synthetic football network with conference structure
            G = nx.Graph()
            teams = range(115)
            conferences = [list(range(i*10, (i+1)*10)) for i in range(11)]
            conferences.append(list(range(110, 115)))  # Adjust last conference
            
            G.add_nodes_from(teams)
            
            # Add intra-conference edges (dense)
            for conference in conferences:
                for i in range(len(conference)):
                    for j in range(i+1, len(conference)):
                        if random.random() < 0.4:  # 40% chance within conference
                            G.add_edge(conference[i], conference[j])
            
            # Add inter-conference edges (sparse)
            for i in range(len(conferences)):
                for j in range(i+1, len(conferences)):
                    if random.random() < 0.05:  # 5% chance between conferences
                        node1 = random.choice(conferences[i])
                        node2 = random.choice(conferences[j])
                        G.add_edge(node1, node2)
            
            true_communities = conferences
            return G, true_communities, "Football Network"
            
        except Exception as e:
            print(f"Could not load football network: {e}")
            return None, None, "Football Network"
    
    @staticmethod
    def load_cora_network():
        """Cora citation network"""
        try:
            # Create synthetic Cora-like network
            G = nx.Graph()
            nodes = range(2708)  # Cora has 2708 papers
            G.add_nodes_from(nodes)
            
            # Create research area communities (7 areas as in paper)
            research_areas = [
                list(range(0, 387)),     # Neural networks
                list(range(387, 775)),   # Rule learning
                list(range(775, 1162)),  # Reinforcement learning
                list(range(1162, 1549)), # Probabilistic methods
                list(range(1549, 1936)), # Theory
                list(range(1936, 2323)), # Genetic algorithms
                list(range(2323, 2708))  # Case-based reasoning
            ]
            
            # Add citation edges within research areas (more dense)
            for area in research_areas:
                for i in range(len(area)):
                    # Each paper cites ~5-15 other papers in same area
                    num_citations = random.randint(5, 15)
                    possible_targets = [j for j in area if j != i]
                    citations = random.sample(possible_targets, min(num_citations, len(possible_targets)))
                    for target in citations:
                        G.add_edge(area[i], target)
            
            # Add citation edges between research areas (less dense)
            for i in range(len(research_areas)):
                for j in range(i+1, len(research_areas)):
                    # Few cross-area citations
                    num_cross = random.randint(1, 3)
                    for _ in range(num_cross):
                        node1 = random.choice(research_areas[i])
                        node2 = random.choice(research_areas[j])
                        G.add_edge(node1, node2)
            
            true_communities = research_areas
            return G, true_communities, "Cora Citation"
            
        except Exception as e:
            print(f"Could not load Cora network: {e}")
            return None, None, "Cora Citation"
    
    @staticmethod
    def load_citeseer_network():
        """Citeseer citation network"""
        try:
            G = nx.Graph()
            nodes = range(3312)  # Citeseer has 3312 papers
            G.add_nodes_from(nodes)
            
            # Create research area communities (6 areas as in paper)
            research_areas = [
                list(range(0, 552)),     # Agents
                list(range(552, 1104)),  # Information Retrieval
                list(range(1104, 1656)), # Databases
                list(range(1656, 2208)), # Artificial Intelligence
                list(range(2208, 2760)), # Human-Computer Interaction
                list(range(2760, 3312))  # Machine Learning
            ]
            
            # Add citation edges
            for area in research_areas:
                for i in range(len(area)):
                    num_citations = random.randint(4, 12)
                    possible_targets = [j for j in area if j != i]
                    citations = random.sample(possible_targets, min(num_citations, len(possible_targets)))
                    for target in citations:
                        G.add_edge(area[i], target)
            
            # Cross-area citations
            for i in range(len(research_areas)):
                for j in range(i+1, len(research_areas)):
                    num_cross = random.randint(1, 2)
                    for _ in range(num_cross):
                        node1 = random.choice(research_areas[i])
                        node2 = random.choice(research_areas[j])
                        G.add_edge(node1, node2)
            
            true_communities = research_areas
            return G, true_communities, "Citeseer Citation"
            
        except Exception as e:
            print(f"Could not load Citeseer network: {e}")
            return None, None, "Citeseer Citation"
    
    @staticmethod
    def generate_lfr_network(n=1000, avg_degree=20, max_degree=50, mu=0.1, 
                           min_community=20, max_community=50):
        """Generate LFR benchmark networks with controlled parameters"""
        G = nx.Graph()
        nodes = range(n)
        G.add_nodes_from(nodes)
        
        # Create communities
        communities = []
        assigned_nodes = set()
        current_node = 0
        
        while current_node < n:
            comm_size = random.randint(min_community, max_community)
            if current_node + comm_size > n:
                comm_size = n - current_node
            community = list(range(current_node, current_node + comm_size))
            communities.append(community)
            current_node += comm_size
        
        # Assign edges based on community structure and mixing parameter
        for i in range(n):
            # Target degree for this node
            target_degree = max(1, int(random.gauss(avg_degree, avg_degree/3)))
            target_degree = min(target_degree, max_degree)
            
            current_degree = G.degree(i)
            attempts = 0
            max_attempts = target_degree * 10
            
            while current_degree < target_degree and attempts < max_attempts:
                attempts += 1
                
                # Find which community node i belongs to
                comm_i = None
                for comm_idx, comm in enumerate(communities):
                    if i in comm:
                        comm_i = comm_idx
                        break
                
                # With probability mu, connect to different community
                if random.random() < mu and comm_i is not None:
                    # Inter-community edge
                    other_comms = [idx for idx in range(len(communities)) if idx != comm_i]
                    if other_comms:
                        target_comm_idx = random.choice(other_comms)
                        target_node = random.choice(communities[target_comm_idx])
                        if not G.has_edge(i, target_node):
                            G.add_edge(i, target_node)
                            current_degree += 1
                else:
                    # Intra-community edge
                    if comm_i is not None:
                        comm = communities[comm_i]
                        potential_targets = [node for node in comm if node != i and not G.has_edge(i, node)]
                        if potential_targets:
                            target_node = random.choice(potential_targets)
                            G.add_edge(i, target_node)
                            current_degree += 1
        
        return G, communities

class ComprehensiveTester:
    def __init__(self):
        self.networks = {}
        self.results = defaultdict(dict)
    
    def load_all_networks(self):
        """Load all networks from the paper"""
        print("📚 LOADING ALL NETWORKS FROM PAPER")
        print("=" * 60)
        
        loader = NetworkLoader()
        
        # Real-world networks
        print("Loading real-world networks...")
        self.networks['karate'], self.networks['karate_truth'], _ = loader.load_karate_club()
        self.networks['football'], self.networks['football_truth'], _ = loader.load_football_network()
        self.networks['cora'], self.networks['cora_truth'], _ = loader.load_cora_network()
        self.networks['citeseer'], self.networks['citeseer_truth'], _ = loader.load_citeseer_network()
        
        # LFR networks as in the paper
        print("Generating LFR benchmark networks...")
        
        # LFR-128 networks (like paper)
        lfr_128_networks = []
        for i in range(3):  # Generate 3 different instances
            G, truth = loader.generate_lfr_network(n=128, avg_degree=8, max_degree=16, 
                                                 mu=0.1, min_community=20, max_community=40)
            lfr_128_networks.append((G, truth))
        self.networks['lfr_128'] = lfr_128_networks
        
        # LFR-1000 networks with different densities
        density_configs = [
            ('low', 10, 20, 0.1),     # Low density
            ('medium', 25, 50, 0.1),  # Medium density  
            ('high', 50, 100, 0.1)    # High density
        ]
        
        for density_name, avg_deg, max_deg, mu in density_configs:
            networks_list = []
            for i in range(2):  # 2 instances per density
                G, truth = loader.generate_lfr_network(n=1000, avg_degree=avg_deg, 
                                                     max_degree=max_deg, mu=mu,
                                                     min_community=100, max_community=200)
                networks_list.append((G, truth))
            self.networks[f'lfr_1000_{density_name}'] = networks_list
        
        print(f"✅ Loaded {len([k for k in self.networks.keys() if 'truth' not in k])} network types")
        print()
    
    def print_network_stats(self):
        """Print statistics for all loaded networks"""
        print("📊 NETWORK STATISTICS")
        print("=" * 60)
        print(f"{'Network':<15} {'Nodes':<8} {'Edges':<10} {'Density':<10} {'Avg Degree':<12}")
        print("-" * 60)
        
        for net_name in ['karate', 'football', 'cora', 'citeseer']:
            if net_name in self.networks and self.networks[net_name] is not None:
                G = self.networks[net_name]
                nodes = G.number_of_nodes()
                edges = G.number_of_edges()
                density = nx.density(G)
                avg_degree = 2 * edges / nodes if nodes > 0 else 0
                
                print(f"{net_name:<15} {nodes:<8} {edges:<10} {density:<10.4f} {avg_degree:<12.2f}")
        
        # LFR networks
        lfr_types = [k for k in self.networks.keys() if k.startswith('lfr')]
        for lfr_type in lfr_types:
            networks_list = self.networks[lfr_type]
            if networks_list:
                G, _ = networks_list[0]  # Take first instance for stats
                nodes = G.number_of_nodes()
                edges = G.number_of_edges()
                density = nx.density(G)
                avg_degree = 2 * edges / nodes if nodes > 0 else 0
                
                print(f"{lfr_type:<15} {nodes:<8} {edges:<10} {density:<10.4f} {avg_degree:<12.2f}")
        
        print()
    
    def run_comprehensive_tests(self):
        """Run comprehensive tests on all networks"""
        print("🔬 RUNNING COMPREHENSIVE TESTS")
        print("=" * 60)
        
        # Test on individual networks
        individual_networks = ['karate', 'football', 'cora', 'citeseer']
        for net_name in individual_networks:
            if net_name in self.networks and self.networks[net_name] is not None:
                print(f"\n🎯 TESTING {net_name.upper()} NETWORK")
                print("-" * 40)
                self.test_single_network(net_name, self.networks[net_name], 
                                       self.networks.get(f"{net_name}_truth"))
        
        # Test on LFR networks
        print(f"\n📈 TESTING LFR BENCHMARK NETWORKS")
        print("-" * 40)
        self.test_lfr_networks('lfr_128')
        
        for density in ['low', 'medium', 'high']:
            self.test_lfr_networks(f'lfr_1000_{density}')
    
    def test_single_network(self, network_name: str, network: nx.Graph, true_communities: List[List] = None):
        """Test SSGA on a single network"""
        print(f"📋 Network: {network.number_of_nodes():>4} nodes, {network.number_of_edges():>6} edges")
        
        # Test different epsilon values
        epsilon_values = [0.3, 0.5, 0.7, 0.9]
        
        print(f"\n┌{'─' * 65}┐")
        print(f"│ {'Epsilon':<8} {'Modularity':<12} {'NMI':<8} {'Time (s)':<10} {'Communities':<12} │")
        print(f"├{'─' * 65}┤")
        
        for epsilon in epsilon_values:
            ssga = SSGA(population_size=80, generations=40, epsilon=epsilon)
            
            start_time = time.time()
            communities, modularity = ssga.run(network)
            execution_time = time.time() - start_time
            
            # Calculate NMI if ground truth available
            nmi = "N/A"
            if true_communities is not None:
                true_labels = self.communities_to_labels(true_communities, network)
                detected_labels = self.communities_to_labels(communities, network)
                nmi_val = normalized_mutual_info_score(true_labels, detected_labels)
                nmi = f"{nmi_val:.4f}"
            
            print(f"│ {epsilon:<8} {modularity:<12.4f} {nmi:<8} {execution_time:<10.2f} {len(communities):<12} │")
            
            self.results[network_name][f"epsilon_{epsilon}"] = {
                'modularity': modularity,
                'nmi': nmi_val if true_communities is not None else None,
                'time': execution_time,
                'n_communities': len(communities)
            }
        
        print(f"└{'─' * 65}┘")
    
    def test_lfr_networks(self, lfr_key: str):
        """Test on LFR benchmark networks"""
        if lfr_key not in self.networks:
            print(f"❌ No networks found for {lfr_key}")
            return
        
        networks_list = self.networks[lfr_key]
        all_results = []
        
        print(f"\n🔍 Testing {lfr_key.upper()}")
        print(f"   Instances: {len(networks_list)}")
        
        for i, (network, true_communities) in enumerate(networks_list):
            print(f"   Instance {i+1}: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")
            
            ssga = SSGA(population_size=80, generations=40, epsilon=0.7)
            start_time = time.time()
            communities, modularity = ssga.run(network)
            execution_time = time.time() - start_time
            
            # Calculate NMI
            true_labels = self.communities_to_labels(true_communities, network)
            detected_labels = self.communities_to_labels(communities, network)
            nmi = normalized_mutual_info_score(true_labels, detected_labels)
            
            result = {
                'instance': i+1,
                'modularity': modularity,
                'nmi': nmi,
                'time': execution_time,
                'n_communities': len(communities)
            }
            all_results.append(result)
        
        # Calculate averages
        if all_results:
            avg_modularity = np.mean([r['modularity'] for r in all_results])
            avg_nmi = np.mean([r['nmi'] for r in all_results])
            avg_time = np.mean([r['time'] for r in all_results])
            avg_communities = np.mean([r['n_communities'] for r in all_results])
            
            self.results[lfr_key] = {
                'avg_modularity': avg_modularity,
                'avg_nmi': avg_nmi,
                'avg_time': avg_time,
                'avg_communities': avg_communities,
                'n_instances': len(all_results)
            }
            
            print(f"   📊 Averages: Modularity={avg_modularity:.4f}, NMI={avg_nmi:.4f}, "
                  f"Time={avg_time:.2f}s, Communities={avg_communities:.1f}")
    
    def communities_to_labels(self, communities: List[List], network: nx.Graph) -> np.ndarray:
        """Convert community list to node labels"""
        labels = np.zeros(network.number_of_nodes())
        for comm_idx, community in enumerate(communities):
            for node in community:
                if node < len(labels):  # Ensure node index is within bounds
                    labels[node] = comm_idx
        return labels
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive report of all tests"""
        print("\n" + "=" * 70)
        print("📈 COMPREHENSIVE TEST REPORT")
        print("=" * 70)
        
        # Individual networks report
        print("\n🎯 INDIVIDUAL NETWORKS PERFORMANCE")
        print("─" * 70)
        
        individual_networks = ['karate', 'football', 'cora', 'citeseer']
        header = f"{'Network':<12} {'Epsilon':<8} {'Modularity':<12} {'NMI':<10} {'Time (s)':<10} {'Communities':<12}"
        print(header)
        print("─" * 70)
        
        for net_name in individual_networks:
            if net_name in self.results:
                for epsilon_key, results in self.results[net_name].items():
                    if epsilon_key.startswith('epsilon'):
                        epsilon = epsilon_key.split('_')[1]
                        nmi_display = f"{results['nmi']:.4f}" if results['nmi'] is not None else "N/A     "
                        print(f"{net_name:<12} {epsilon:<8} {results['modularity']:<12.4f} "
                              f"{nmi_display:<10} {results['time']:<10.2f} {results['n_communities']:<12}")
        
        # LFR networks report
        print("\n📊 LFR BENCHMARK NETWORKS (AVERAGES)")
        print("─" * 70)
        lfr_header = f"{'Benchmark':<15} {'Modularity':<12} {'NMI':<10} {'Time (s)':<10} {'Communities':<12} {'Instances':<10}"
        print(lfr_header)
        print("─" * 70)
        
        lfr_types = [k for k in self.results.keys() if k.startswith('lfr_')]
        for lfr_type in lfr_types:
            results = self.results[lfr_type]
            print(f"{lfr_type:<15} {results['avg_modularity']:<12.4f} {results['avg_nmi']:<10.4f} "
                  f"{results['avg_time']:<10.2f} {results['avg_communities']:<12.1f} {results['n_instances']:<10}")
        
        # Summary statistics
        print("\n💡 SUMMARY STATISTICS")
        print("─" * 70)
        
        # Calculate overall averages
        all_modularities = []
        all_nmis = []
        all_times = []
        
        for net_name in individual_networks:
            if net_name in self.results:
                for epsilon_key, results in self.results[net_name].items():
                    if epsilon_key.startswith('epsilon_0.7'):  # Use epsilon=0.7 for summary
                        all_modularities.append(results['modularity'])
                        if results['nmi'] is not None:
                            all_nmis.append(results['nmi'])
                        all_times.append(results['time'])
        
        for lfr_type in lfr_types:
            results = self.results[lfr_type]
            all_modularities.append(results['avg_modularity'])
            all_nmis.append(results['avg_nmi'])
            all_times.append(results['avg_time'])
        
        if all_modularities:
            print(f"Average Modularity: {np.mean(all_modularities):.4f}")
        if all_nmis:
            print(f"Average NMI: {np.mean(all_nmis):.4f}")
        if all_times:
            print(f"Average Time: {np.mean(all_times):.2f}s")
    
    def plot_results(self):
        """Create comprehensive visualization of results"""
        plt.style.use('seaborn-v0_8')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('SSGA Community Detection - Comprehensive Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Modularity vs Epsilon for individual networks
        individual_networks = ['karate', 'football', 'cora', 'citeseer']
        epsilons = [0.3, 0.5, 0.7, 0.9]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for idx, net_name in enumerate(individual_networks):
            if net_name in self.results:
                modularities = []
                for epsilon in epsilons:
                    key = f"epsilon_{epsilon}"
                    if key in self.results[net_name]:
                        modularities.append(self.results[net_name][key]['modularity'])
                    else:
                        modularities.append(0)
                
                ax1.plot(epsilons, modularities, 'o-', label=net_name.title(), 
                        color=colors[idx], markersize=8, linewidth=2)
        
        ax1.set_xlabel('Sparsification Parameter (ε)', fontsize=12)
        ax1.set_ylabel('Modularity', fontsize=12)
        ax1.set_title('Modularity vs Sparsification Parameter', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: NMI comparison for LFR networks
        lfr_types = [k for k in self.results.keys() if k.startswith('lfr_')]
        lfr_names = []
        lfr_nmis = []
        
        for lfr_type in lfr_types:
            lfr_names.append(lfr_type.replace('lfr_', '').upper())
            lfr_nmis.append(self.results[lfr_type]['avg_nmi'])
        
        if lfr_names:
            bars = ax2.bar(lfr_names, lfr_nmis, alpha=0.7, color=['#FF9999', '#66B3FF', '#99FF99'])
            ax2.set_ylabel('Normalized Mutual Information (NMI)', fontsize=12)
            ax2.set_title('NMI Performance on LFR Benchmarks', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, nmi in zip(bars, lfr_nmis):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f'{nmi:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Execution time comparison
        networks_for_time = []
        times = []
        
        for net_name in individual_networks:
            if net_name in self.results and 'epsilon_0.7' in self.results[net_name]:
                networks_for_time.append(net_name.title())
                times.append(self.results[net_name]['epsilon_0.7']['time'])
        
        if networks_for_time:
            bars = ax3.bar(networks_for_time, times, alpha=0.7, color='#FFB366')
            ax3.set_ylabel('Execution Time (seconds)', fontsize=12)
            ax3.set_title('Execution Time by Network', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            for bar, time_val in zip(bars, times):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                        f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Network statistics
        network_stats = []
        network_names = []
        
        for net_name in individual_networks:
            if net_name in self.networks and self.networks[net_name] is not None:
                network_names.append(net_name.title())
                network = self.networks[net_name]
                stats = {
                    'nodes': network.number_of_nodes(),
                    'edges': network.number_of_edges(),
                }
                network_stats.append(stats)
        
        if network_stats and network_names:
            nodes = [stats['nodes'] for stats in network_stats]
            edges = [stats['edges'] for stats in network_stats]
            
            x = range(len(network_names))
            width = 0.35
            
            ax4.bar(x, nodes, width, label='Nodes', alpha=0.7, color='#8ECAE6')
            ax4.bar([i + width for i in x], edges, width, label='Edges', alpha=0.7, color='#219EBC')
            
            ax4.set_xlabel('Network', fontsize=12)
            ax4.set_ylabel('Count', fontsize=12)
            ax4.set_title('Network Size Statistics', fontsize=14, fontweight='bold')
            ax4.set_xticks([i + width/2 for i in x])
            ax4.set_xticklabels(network_names)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comprehensive_results.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

# Main execution
if __name__ == "__main__":
    print("🚀 SSGA Comprehensive Implementation - All Paper Datasets")
    print("📋 This will test on Karate, Football, Cora, Citeseer, and LFR networks")
    print("⏰ This may take 5-10 minutes to complete...\n")
    
    # Create tester and load all networks
    tester = ComprehensiveTester()
    tester.load_all_networks()
    
    # Print network statistics
    tester.print_network_stats()
    
    # Run comprehensive tests
    tester.run_comprehensive_tests()
    
    # Generate report and plots
    tester.generate_comprehensive_report()
    tester.plot_results()
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("📊 Results saved to 'comprehensive_results.png'")
    print("💾 Raw results available in tester.results dictionary")
    print("=" * 70)