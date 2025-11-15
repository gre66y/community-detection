import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import SpectralClustering
import requests
import zipfile
import io
import os

class CoraCommunityDetection:
    def __init__(self):
        self.G = None
        self.ground_truth = None
        
    def download_and_load_cora_data(self):
        """Download and load Cora citation dataset"""
        print("Loading Cora citation network...")
        
        try:
            # Cora dataset URL
            url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
            
            # Download and extract
            print("Downloading Cora dataset...")
            response = requests.get(url)
            response.raise_for_status()
            
            # Extract tar.gz file
            import tarfile
            with tarfile.open(fileobj=io.BytesIO(response.content), mode='r:gz') as tar:
                # Extract to temporary directory
                tar.extractall('cora_data')
            
            # Load the data
            self.G = nx.Graph()
            self.ground_truth = {}
            
            # Read node labels (ground truth)
            label_file = 'cora_data/cora/cora.content'
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        node_id = int(parts[0])
                        # Last column is the class label
                        class_label = parts[-1]
                        self.ground_truth[node_id] = class_label
                        self.G.add_node(node_id, label=class_label)
            
            # Read edges (citations)
            cites_file = 'cora_data/cora/cora.cites'
            if os.path.exists(cites_file):
                with open(cites_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) == 2:
                            source = int(parts[0])
                            target = int(parts[1])
                            self.G.add_edge(source, target)
            
            print(f"Successfully loaded Cora network:")
            print(f"  - Nodes: {self.G.number_of_nodes()}")
            print(f"  - Edges: {self.G.number_of_edges()}")
            print(f"  - Ground truth communities: {len(set(self.ground_truth.values()))}")
            
        except Exception as e:
            print(f"Error downloading/loading Cora data: {e}")
            print("Creating synthetic Cora-like network...")
            self.create_synthetic_cora()
    
    def create_synthetic_cora(self):
        """Create a synthetic Cora-like network if download fails"""
        self.G = nx.Graph()
        self.ground_truth = {}
        
        # Research areas in Cora
        research_areas = {
            'Neural_Networks': 0,
            'Rule_Learning': 1,
            'Reinforcement_Learning': 2,
            'Probabilistic_Methods': 3,
            'Theory': 4,
            'Genetic_Algorithms': 5,
            'Case_Based': 6
        }
        
        # Create 2708 nodes (like real Cora)
        num_nodes = 2708
        nodes_per_area = num_nodes // len(research_areas)
        
        node_id = 0
        for area, area_id in research_areas.items():
            for i in range(nodes_per_area):
                self.G.add_node(node_id)
                self.ground_truth[node_id] = area_id
                node_id += 1
        
        # Add remaining nodes
        while node_id < num_nodes:
            self.G.add_node(node_id)
            self.ground_truth[node_id] = np.random.randint(0, len(research_areas))
            node_id += 1
        
        # Add edges - more within communities, fewer between
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                # Same research area - higher probability of connection
                if self.ground_truth[i] == self.ground_truth[j]:
                    if np.random.random() < 0.02:  # 2% within communities
                        self.G.add_edge(i, j)
                else:
                    if np.random.random() < 0.001:  # 0.1% between communities
                        self.G.add_edge(i, j)
        
        print(f"Created synthetic Cora network:")
        print(f"  - Nodes: {self.G.number_of_nodes()}")
        print(f"  - Edges: {self.G.number_of_edges()}")
        print(f"  - Ground truth communities: {len(set(self.ground_truth.values()))}")
    
    def detect_communities_girvan_newman(self, k=7):
        """Detect communities using Girvan-Newman algorithm"""
        print("Detecting communities with Girvan-Newman algorithm...")
        print("Note: This may take a while for large networks...")
        
        # Use a subgraph for faster computation if network is too large
        if self.G.number_of_nodes() > 1000:
            print("Using largest connected component for faster computation...")
            largest_cc = max(nx.connected_components(self.G), key=len)
            G_sub = self.G.subgraph(largest_cc).copy()
        else:
            G_sub = self.G
        
        comp = nx.algorithms.community.girvan_newman(G_sub)
        
        communities = []
        for i, communities_at_level in enumerate(comp):
            if len(communities_at_level) >= k or i >= 3:  # Limit iterations for speed
                communities = list(communities_at_level)
                break
        
        partition = {}
        for comm_id, community in enumerate(communities):
            for node in community:
                partition[node] = comm_id
        
        # Map back to original graph if we used subgraph
        if self.G.number_of_nodes() > 1000:
            full_partition = {}
            for node in self.G.nodes():
                if node in partition:
                    full_partition[node] = partition[node]
                else:
                    full_partition[node] = -1  # Unassigned
            return full_partition
        
        return partition
    
    def detect_communities_spectral(self, k=7):
        """Detect communities using spectral clustering"""
        print("Detecting communities with Spectral Clustering...")
        
        # Use largest connected component for spectral clustering
        largest_cc = max(nx.connected_components(self.G), key=len)
        G_sub = self.G.subgraph(largest_cc).copy()
        
        nodes = list(G_sub.nodes())
        adj_matrix = nx.to_numpy_array(G_sub, nodelist=nodes)
        
        sc = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=42)
        labels = sc.fit_predict(adj_matrix)
        
        partition = {node: label for node, label in zip(nodes, labels)}
        
        # Map back to original graph
        full_partition = {}
        for node in self.G.nodes():
            if node in partition:
                full_partition[node] = partition[node]
            else:
                full_partition[node] = -1  # Unassigned for disconnected nodes
        
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
    
    def visualize_communities(self, detected_communities, title="Cora Citation Network Communities"):
        """Visualize the network with detected communities"""
        # Use a sample of nodes for visualization (for performance)
        if self.G.number_of_nodes() > 500:
            print("Visualizing a sample of 500 nodes for clarity...")
            sample_nodes = list(self.G.nodes())[:500]
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
            
        plt.figure(figsize=(15, 10))
        
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
                    alpha=0.7,
                    label=f'Community {comm_id}'
                )
        
        # Draw edges
        nx.draw_networkx_edges(G_viz, pos, alpha=0.2, width=0.5)
        
        plt.title(f"{title}\n(Sample of {G_viz.number_of_nodes()} nodes)")
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def visualize_ground_truth(self):
        """Visualize the ground truth communities"""
        # Use a sample for visualization
        if self.G.number_of_nodes() > 500:
            sample_nodes = list(self.G.nodes())[:500]
            G_viz = self.G.subgraph(sample_nodes)
        else:
            G_viz = self.G
            
        plt.figure(figsize=(15, 10))
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
                alpha=0.7,
                label=f'Research Area {comm_id}'
            )
        
        nx.draw_networkx_edges(G_viz, pos, alpha=0.2, width=0.5)
        
        plt.title(f"Cora Network - Ground Truth Research Areas\n(Sample of {G_viz.number_of_nodes()} nodes)")
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline with both algorithms"""
        # Load data
        self.download_and_load_cora_data()
        
        print("\n" + "="*60)
        print("COMMUNITY DETECTION ANALYSIS - CORA CITATION NETWORK")
        print("="*60)
        
        # Show ground truth first
        self.visualize_ground_truth()
        
        results = {}
        
        # Test both algorithms
        algorithms = [
            ("Girvan-Newman", lambda: self.detect_communities_girvan_newman(k=7)),
            ("Spectral Clustering", lambda: self.detect_communities_spectral(k=7))
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
                
                # Visualize results for each algorithm
                self.visualize_communities(communities, f"{algo_name} Communities")
                    
            except Exception as e:
                print(f"Error with {algo_name}: {e}")
                continue
        
        # Print comparison summary
        self.print_comparison_summary(results)
        
        return results
    
    def print_comparison_summary(self, results):
        """Print a comparison summary of both algorithms"""
        print("\n" + "="*60)
        print("COMPARISON SUMMARY - CORA DATASET")
        print("="*60)
        print(f"{'Algorithm':<20} {'NMI':<10} {'Modularity':<12} {'Communities':<12} {'Nodes Covered':<15}")
        print(f"{'-'*70}")
        
        for algo_name, result in results.items():
            if result['nmi'] is not None:
                nodes_covered = sum(1 for comm in result['communities'].values() if comm != -1)
                print(f"{algo_name:<20} {result['nmi']:<10.4f} {result['modularity']:<12.4f} "
                      f"{result['num_communities']:<12} {nodes_covered}/{self.G.number_of_nodes():<15}")
        
        # Find best algorithm
        if results:
            valid_results = {k: v for k, v in results.items() if v['nmi'] is not None}
            if valid_results:
                best_algo = max(valid_results.items(), key=lambda x: x[1]['nmi'])
                print(f"\n🏆 Best Algorithm: {best_algo[0]} (NMI: {best_algo[1]['nmi']:.4f})")

# Run the analysis
if __name__ == "__main__":
    analyzer = CoraCommunityDetection()
    results = analyzer.run_complete_analysis()