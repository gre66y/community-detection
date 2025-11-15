import numpy as np
import networkx as nx
import urllib.request
import io
import zipfile
import random
from typing import List, Tuple, Dict
import time

def download_football_network():
    """Download American College Football network"""
    url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"
    
    with urllib.request.urlopen(url) as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as z:
            # Read the GML file from zip
            with z.open('football.gml') as f:
                content = f.read().decode('utf-8')
                # Parse GML content
                G = nx.parse_gml(content)
    return G

class SSGA:
    def __init__(self, population_size=100, generations=50, crossover_rate=0.9, 
        mutation_rate=0.05, epsilon=0.3):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.epsilon = epsilon
    
    def compute_effective_resistance(self, graph: nx.Graph) -> Dict[Tuple, float]:
        n = graph.number_of_nodes()
        nodes = list(graph.nodes())
    
        # Create adjacency matrix - keep it as sparse initially
        A = nx.adjacency_matrix(graph, nodelist=nodes)
        
        # Compute Laplacian matrix: L = D - A
        D = np.diag(np.array(A.sum(axis=1)).flatten())  # Convert to dense diagonal matrix
        L = D - A.todense()  # Convert to dense for pseudoinverse
        
        # Compute pseudoinverse of Laplacian
        try:
            L_plus = np.linalg.pinv(L)
        except:
            # Fallback: use regularized version if matrix is singular
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
        effective_resistances = self.compute_effective_resistance(graph)
        
        # Compute sampling probabilities
        total_resistance = sum(effective_resistances.values())
        sampling_probs = {}
        
        for edge, resistance in effective_resistances.items():
            sampling_probs[edge] = resistance / total_resistance
        
        # Number of samples (simplified version)
        q = int(8 * n * np.log(n) / (self.epsilon ** 2))
        q = min(q, graph.number_of_edges())  # Don't sample more than available edges
        
        # Sample edges
        edges = list(graph.edges())
        probabilities = [sampling_probs[edge] for edge in edges]
        
        # Normalize probabilities
        prob_sum = sum(probabilities)
        probabilities = [p / prob_sum for p in probabilities]
        
        # Create sparsified graph
        sparsified_graph = nx.Graph()
        sparsified_graph.add_nodes_from(graph.nodes())
        
        # Sample edges with replacement
        sampled_indices = np.random.choice(len(edges), size=q, p=probabilities, replace=True)
        
        for idx in sampled_indices:
            u, v = edges[idx]
            if sparsified_graph.has_edge(u, v):
                # Increase weight if edge already exists
                sparsified_graph[u][v]['weight'] += 1
            else:
                sparsified_graph.add_edge(u, v, weight=1)
                
        return sparsified_graph
    
    def initialize_population(self, graph: nx.Graph) -> List[List[int]]:
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
        print("Step 1: Spectral sparsification...")
        sparsified_graph = self.spectral_sparsification(graph)
        print(f"Original edges: {graph.number_of_edges()}, Sparsified edges: {sparsified_graph.number_of_edges()}")
        
        print("Step 2: Genetic algorithm for community detection...")
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
                print(f"Generation {generation}, Best Fitness: {best_fitness:.4f}")
        
        return best_communities, best_fitness

def analyze_football_network():
    """Analyze the football network using SSGA"""
    print("=== American College Football Network Analysis ===")
    
    # Download and load the football network
    print("Downloading football network...")
    G = download_football_network()
    
    print(f"Network Information:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    
    # Get ground truth communities (conferences)
    conferences = {}
    for node, data in G.nodes(data=True):
        conference = data.get('value', 'Unknown')
        if conference not in conferences:
            conferences[conference] = []
        conferences[conference].append(node)
    
    true_communities = list(conferences.values())
    print(f"Number of conferences (ground truth communities): {len(true_communities)}")
    
    # Display conference sizes
    print("\nConference sizes:")
    for conference, teams in conferences.items():
        print(f"Conference {conference}: {len(teams)} teams")
    
    # Run SSGA
    ssga = SSGA(population_size=50, generations=30, epsilon=0.6)
    start_time = time.time()
    detected_communities, modularity_score = ssga.run(G)
    end_time = time.time()
    
    print(f"\nSSGA Results:")
    print(f"Detected {len(detected_communities)} communities")
    print(f"Modularity: {modularity_score:.4f}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    
    # Display detected community sizes
    print("\nDetected community sizes:")
    for i, community in enumerate(detected_communities):
        print(f"Community {i+1}: {len(community)} nodes")
    
    # Compare with ground truth using NMI
    from sklearn.metrics import normalized_mutual_info_score
    
    # Convert to label format for NMI calculation
    node_list = list(G.nodes())
    
    true_labels = np.zeros(len(node_list))
    for comm_idx, community in enumerate(true_communities):
        for node in community:
            true_labels[node_list.index(node)] = comm_idx
    
    detected_labels = np.zeros(len(node_list))
    for comm_idx, community in enumerate(detected_communities):
        for node in community:
            detected_labels[node_list.index(node)] = comm_idx
    
    # Calculate NMI
    nmi = normalized_mutual_info_score(true_labels, detected_labels)
    print(f"\nNormalized Mutual Information (NMI) with ground truth conferences: {nmi:.4f}")
    
    return detected_communities, modularity_score, nmi, true_communities

def compare_with_louvain_football():
    """Compare SSGA with Louvain algorithm on football network"""
    try:
        from community import community_louvain
        
        print("\n=== Comparison with Louvain Algorithm ===")
        
        # Download football network
        G = download_football_network()
        
        # Louvain algorithm
        start_time = time.time()
        louvain_partition = community_louvain.best_partition(G)
        louvain_time = time.time() - start_time
        
        # Convert partition to communities format
        louvain_communities = {}
        for node, community_id in louvain_partition.items():
            if community_id not in louvain_communities:
                louvain_communities[community_id] = []
            louvain_communities[community_id].append(node)
        
        louvain_communities_list = list(louvain_communities.values())
        louvain_modularity = community_louvain.modularity(louvain_partition, G)
        
        print(f"Louvain - Communities: {len(louvain_communities_list)}, Modularity: {louvain_modularity:.4f}, Time: {louvain_time:.2f}s")
        
        # SSGA
        ssga = SSGA(population_size=50, generations=30)
        start_time = time.time()
        ssga_communities, ssga_modularity = ssga.run(G)
        ssga_time = time.time() - start_time
        
        print(f"SSGA - Communities: {len(ssga_communities)}, Modularity: {ssga_modularity:.4f}, Time: {ssga_time:.2f}s")
        
        return louvain_communities_list, louvain_modularity, ssga_communities, ssga_modularity
        
    except ImportError:
        print("python-louvain package not installed. Install with: pip install python-louvain")
        return None, None, None, None

if __name__ == "__main__":
    # Install required packages first:
    # pip install networkx numpy scikit-learn python-louvain
    
    print("SSGA Community Detection on American College Football Network")
    print("=" * 70)
    
    # Run SSGA on football network
    detected_communities, mod_score, nmi, true_communities = analyze_football_network()
    
    # Compare with Louvain
    compare_with_louvain_football()
    
    print("\nAnalysis complete!")