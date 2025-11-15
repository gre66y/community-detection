import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import SpectralClustering
import networkx.algorithms.community as nx_comm

class LFRCommunityDetection:
    def __init__(self):
        self.G = None
        self.ground_truth = None
        self.lfr_params = None
        
    def generate_lfr_network(self, num_nodes=1000, mixing_param=0.3, network_type="small"):
        """Generate LFR benchmark networks with different parameters"""
        print("Generating LFR benchmark network...")
        
        if network_type == "small":
            # Small network parameters (like LFR-128)
            params = {
                'n': 128,
                'tau1': 2,        # Power law exponent for degree distribution
                'tau2': 1,        # Power law exponent for community size distribution  
                'mu': mixing_param, # Mixing parameter
                'average_degree': 8,
                'min_degree': 2,
                'max_degree': 9,
                'min_community': 20,
                'max_community': 40
            }
        else:
            # Large network parameters (like LFR-1000)
            params = {
                'n': num_nodes,
                'tau1': 2,
                'tau2': 2,
                'mu': mixing_param,
                'average_degree': 20,
                'min_degree': 5,
                'max_degree': 50,
                'min_community': 50,
                'max_community': 150
            }
        
        self.lfr_params = params
        
        try:
            # Generate LFR benchmark graph
            self.G = nx.LFR_benchmark_graph(**params, seed=42)
            
            # Extract ground truth communities
            self.ground_truth = {}
            for node, data in self.G.nodes(data=True):
                if 'community' in data:
                    # LFR stores communities as sets
                    if data['community']:
                        self.ground_truth[node] = list(data['community'])[0]
                    else:
                        self.ground_truth[node] = -1
                else:
                    self.ground_truth[node] = -1
            
            # Remove nodes without communities
            valid_nodes = [node for node, comm in self.ground_truth.items() if comm != -1]
            self.G = self.G.subgraph(valid_nodes)
            self.ground_truth = {node: comm for node, comm in self.ground_truth.items() if comm != -1}
            
            print(f"Generated LFR network:")
            print(f"  - Nodes: {self.G.number_of_nodes()}")
            print(f"  - Edges: {self.G.number_of_edges()}")
            print(f"  - Ground truth communities: {len(set(self.ground_truth.values()))}")
            print(f"  - Mixing parameter (μ): {mixing_param}")
            print(f"  - Network density: {nx.density(self.G):.4f}")
            
        except Exception as e:
            print(f"Error generating LFR network: {e}")
            print("Creating synthetic network with similar properties...")
            self.create_synthetic_network(params)
    
    def create_synthetic_network(self, params):
        """Create synthetic network if LFR generation fails"""
        n = params['n']
        mu = params['mu']
        
        self.G = nx.Graph()
        self.ground_truth = {}
        
        # Create communities
        num_communities = max(3, n // 50)  # Rough estimate
        communities = {}
        nodes_per_comm = n // num_communities
        
        node_id = 0
        for comm_id in range(num_communities):
            comm_nodes = list(range(node_id, min(node_id + nodes_per_comm, n)))
            communities[comm_id] = comm_nodes
            for node in comm_nodes:
                self.G.add_node(node)
                self.ground_truth[node] = comm_id
            node_id += nodes_per_comm
        
        # Add remaining nodes
        while node_id < n:
            self.G.add_node(node_id)
            comm_id = np.random.randint(0, num_communities)
            self.ground_truth[node_id] = comm_id
            communities[comm_id].append(node_id)
            node_id += 1
        
        # Add edges based on mixing parameter
        for i in range(n):
            for j in range(i + 1, n):
                if self.ground_truth[i] == self.ground_truth[j]:
                    # Within community - higher probability
                    if np.random.random() < (1 - mu) * 0.3:
                        self.G.add_edge(i, j)
                else:
                    # Between communities - lower probability  
                    if np.random.random() < mu * 0.05:
                        self.G.add_edge(i, j)
        
        print(f"Created synthetic network:")
        print(f"  - Nodes: {self.G.number_of_nodes()}")
        print(f"  - Edges: {self.G.number_of_edges()}")
        print(f"  - Communities: {len(set(self.ground_truth.values()))}")
    
    def detect_communities_girvan_newman(self, k=None):
        """Detect communities using Girvan-Newman algorithm"""
        print("Detecting communities with Girvan-Newman algorithm...")
        
        if k is None:
            k = len(set(self.ground_truth.values()))
        
        # Use largest connected component for better results
        largest_cc = max(nx.connected_components(self.G), key=len)
        G_sub = self.G.subgraph(largest_cc).copy()
        
        print(f"Working on subgraph with {G_sub.number_of_nodes()} nodes...")
        
        comp = nx.algorithms.community.girvan_newman(G_sub)
        
        communities = []
        max_iterations = 5
        
        for i, communities_at_level in enumerate(comp):
            current_k = len(communities_at_level)
            print(f"Iteration {i}: Found {current_k} communities")
            
            if current_k >= k or i >= max_iterations:
                communities = list(communities_at_level)
                break
        
        partition = {}
        for comm_id, community in enumerate(communities):
            for node in community:
                partition[node] = comm_id
        
        # Map back to original graph
        full_partition = {}
        for node in self.G.nodes():
            if node in partition:
                full_partition[node] = partition[node]
            else:
                full_partition[node] = -1
        
        return full_partition
    
    def detect_communities_spectral(self, k=None):
        """Detect communities using spectral clustering"""
        print("Detecting communities with Spectral Clustering...")
        
        if k is None:
            k = len(set(self.ground_truth.values()))
        
        # Use largest connected component for spectral clustering
        largest_cc = max(nx.connected_components(self.G), key=len)
        G_sub = self.G.subgraph(largest_cc).copy()
        
        print(f"Working on subgraph with {G_sub.number_of_nodes()} nodes...")
        
        nodes = list(G_sub.nodes())
        adj_matrix = nx.to_numpy_array(G_sub, nodelist=nodes)
        
        print("Running spectral clustering...")
        sc = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=42)
        labels = sc.fit_predict(adj_matrix)
        
        partition = {node: label for node, label in zip(nodes, labels)}
        
        # Map back to original graph
        full_partition = {}
        for node in self.G.nodes():
            if node in partition:
                full_partition[node] = partition[node]
            else:
                full_partition[node] = -1
        
        return full_partition
    
    def evaluate_communities(self, detected_communities, algorithm_name):
        """Evaluate the detected communities against ground truth"""
        # Filter out unassigned nodes (-1)
        valid_nodes = {node: comm for node, comm in detected_communities.items() 
                      if comm != -1 and node in self.ground_truth}
        
        if not valid_nodes:
            print("No valid communities detected for evaluation")
            return None, None
        
        # Calculate NMI (Normalized Mutual Information)
        true_labels = []
        detected_labels = []
        
        for node, true_comm in self.ground_truth.items():
            if node in valid_nodes:
                true_labels.append(true_comm)
                detected_labels.append(valid_nodes[node])
        
        if not true_labels:
            print("No overlapping nodes for evaluation")
            return None, None
        
        nmi = normalized_mutual_info_score(true_labels, detected_labels)
        
        # Calculate modularity on valid subgraph
        valid_subgraph = self.G.subgraph(valid_nodes.keys())
        communities_dict = defaultdict(list)
        for node, comm in valid_nodes.items():
            communities_dict[comm].append(node)
        
        communities_list = [set(nodes) for nodes in communities_dict.values()]
        modularity = nx.algorithms.community.quality.modularity(valid_subgraph, communities_list)
        
        print(f"\n{algorithm_name} Results:")
        print(f"  - Number of communities detected: {len(set(valid_nodes.values()))}")
        print(f"  - Ground truth communities: {len(set(self.ground_truth.values()))}")
        print(f"  - Nodes assigned to communities: {len(valid_nodes)}/{self.G.number_of_nodes()}")
        print(f"  - Normalized Mutual Information (NMI): {nmi:.4f}")
        print(f"  - Modularity: {modularity:.4f}")
        
        return nmi, modularity
    
    def visualize_communities(self, detected_communities, title="LFR Network Communities"):
        """Visualize the network with detected communities"""
        # Use a sample if network is too large
        sample_size = min(300, self.G.number_of_nodes())
        if self.G.number_of_nodes() > sample_size:
            print(f"Visualizing a sample of {sample_size} nodes for clarity...")
            sample_nodes = list(self.G.nodes())[:sample_size]
            G_viz = self.G.subgraph(sample_nodes)
            viz_partition = {node: detected_communities[node] for node in sample_nodes 
                           if detected_communities[node] != -1}
        else:
            G_viz = self.G
            viz_partition = {node: comm for node, comm in detected_communities.items() 
                           if comm != -1}
        
        if not viz_partition:
            print("No communities to visualize")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Create layout
        pos = nx.spring_layout(G_viz, seed=42, k=0.3, iterations=50)
        
        # Get community colors
        communities = defaultdict(list)
        for node, comm_id in viz_partition.items():
            if node in G_viz:
                communities[comm_id].append(node)
        
        # Plot each community with different color
        colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))
        for i, (comm_id, nodes) in enumerate(communities.items()):
            valid_nodes = [node for node in nodes if node in G_viz]
            if valid_nodes:
                nx.draw_networkx_nodes(
                    G_viz, pos, 
                    nodelist=valid_nodes,
                    node_color=[colors[i]],
                    node_size=50,
                    alpha=0.8,
                    label=f'Community {comm_id}'
                )
        
        # Draw edges
        nx.draw_networkx_edges(G_viz, pos, alpha=0.3, width=0.5)
        
        plt.title(f"{title}\n(Sample of {G_viz.number_of_nodes()} nodes)")
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def visualize_ground_truth(self):
        """Visualize the ground truth communities"""
        sample_size = min(300, self.G.number_of_nodes())
        if self.G.number_of_nodes() > sample_size:
            sample_nodes = list(self.G.nodes())[:sample_size]
            G_viz = self.G.subgraph(sample_nodes)
        else:
            G_viz = self.G
            
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G_viz, seed=42, k=0.3, iterations=50)
        
        # Group nodes by ground truth
        communities_gt = defaultdict(list)
        for node in G_viz.nodes():
            if node in self.ground_truth:
                communities_gt[self.ground_truth[node]].append(node)
        
        # Plot
        colors = plt.cm.Set3(np.linspace(0, 1, len(communities_gt)))
        for i, (comm_id, nodes) in enumerate(communities_gt.items()):
            nx.draw_networkx_nodes(
                G_viz, pos, 
                nodelist=nodes,
                node_color=[colors[i]],
                node_size=50,
                alpha=0.8,
                label=f'True Community {comm_id}'
            )
        
        nx.draw_networkx_edges(G_viz, pos, alpha=0.3, width=0.5)
        
        plt.title(f"LFR Network - Ground Truth Communities\n(Sample of {G_viz.number_of_nodes()} nodes)")
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def run_single_experiment(self, num_nodes=1000, mixing_param=0.3, network_type="large"):
        """Run a single experiment with given parameters"""
        print("\n" + "="*60)
        print(f"LFR EXPERIMENT: {num_nodes} nodes, μ={mixing_param}")
        print("="*60)
        
        # Generate network
        self.generate_lfr_network(num_nodes, mixing_param, network_type)
        
        # Show ground truth
        self.visualize_ground_truth()
        
        results = {}
        
        # Test both algorithms
        true_k = len(set(self.ground_truth.values()))
        algorithms = [
            ("Girvan-Newman", lambda: self.detect_communities_girvan_newman(k=true_k)),
            ("Spectral Clustering", lambda: self.detect_communities_spectral(k=true_k))
        ]
        
        for algo_name, algo_func in algorithms:
            try:
                print(f"\n{'='*50}")
                print(f"Running {algo_name}...")
                print('='*50)
                
                communities = algo_func()
                nmi, modularity = self.evaluate_communities(communities, algo_name)
                
                results[algo_name] = {
                    'communities': communities,
                    'nmi': nmi,
                    'modularity': modularity,
                    'num_communities': len(set(comm for comm in communities.values() if comm != -1))
                }
                
                # Visualize results
                self.visualize_communities(communities, f"{algo_name} Communities")
                    
            except Exception as e:
                print(f"Error with {algo_name}: {e}")
                continue
        
        return results
    
    def run_comprehensive_study(self):
        """Run comprehensive study with different parameters"""
        print("COMPREHENSIVE LFR BENCHMARK STUDY")
        print("Comparing Girvan-Newman vs Spectral Clustering")
        
        all_results = {}
        
        # Test different mixing parameters
        mixing_params = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        for mu in mixing_params:
            print(f"\n\n{'#'*70}")
            print(f"TESTING WITH MIXING PARAMETER μ = {mu}")
            print(f"{'#'*70}")
            
            results = self.run_single_experiment(num_nodes=500, mixing_param=mu, network_type="large")
            all_results[mu] = results
            
            # Print summary for this mu
            self.print_mu_summary(mu, results)
        
        # Final comparison
        self.print_final_comparison(all_results)
        
        return all_results
    
    def print_mu_summary(self, mu, results):
        """Print summary for a specific mixing parameter"""
        print(f"\nμ = {mu} Summary:")
        print(f"{'Algorithm':<20} {'NMI':<10} {'Modularity':<12}")
        print(f"{'-'*40}")
        
        for algo_name, result in results.items():
            if result['nmi'] is not None:
                print(f"{algo_name:<20} {result['nmi']:<10.4f} {result['modularity']:<12.4f}")
    
    def print_final_comparison(self, all_results):
        """Print final comparison across all mixing parameters"""
        print("\n\n" + "="*80)
        print("FINAL COMPARISON: GIRVAN-NEWMAN vs SPECTRAL CLUSTERING")
        print("="*80)
        
        print(f"\n{'Mixing Param (μ)':<15} {'Algorithm':<20} {'NMI':<10} {'Modularity':<12}")
        print(f"{'-'*60}")
        
        gn_nmis = []
        sc_nmis = []
        
        for mu, results in all_results.items():
            for algo_name, result in results.items():
                if result['nmi'] is not None:
                    print(f"{mu:<15} {algo_name:<20} {result['nmi']:<10.4f} {result['modularity']:<12.4f}")
                    
                    if algo_name == "Girvan-Newman":
                        gn_nmis.append(result['nmi'])
                    else:
                        sc_nmis.append(result['nmi'])
        
        if gn_nmis and sc_nmis:
            print(f"\nAverage NMI - Girvan-Newman: {np.mean(gn_nmis):.4f}")
            print(f"Average NMI - Spectral Clustering: {np.mean(sc_nmis):.4f}")
            
            best_avg = "Girvan-Newman" if np.mean(gn_nmis) > np.mean(sc_nmis) else "Spectral Clustering"
            print(f"🏆 Best Overall: {best_avg}")

# Run the analysis
if __name__ == "__main__":
    analyzer = LFRCommunityDetection()
    
    # Run single experiment
    # results = analyzer.run_single_experiment(num_nodes=500, mixing_param=0.3)
    
    # Run comprehensive study
    all_results = analyzer.run_comprehensive_study()