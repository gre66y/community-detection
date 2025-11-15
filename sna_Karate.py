import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import SpectralClustering

class KarateCommunityDetection:
    def __init__(self):
        self.G = None
        self.ground_truth = None
        
    def load_karate_data(self):
        """Load Zachary Karate Club dataset"""
        print("Loading Zachary Karate Club dataset...")
        
        # Load the built-in Karate Club graph from networkx
        self.G = nx.karate_club_graph()
        
        # Extract ground truth (the actual split that happened)
        self.ground_truth = {}
        for node in self.G.nodes():
            if node <= 16:
                self.ground_truth[node] = 0  # Mr. Hi's group
            else:
                self.ground_truth[node] = 1  # John A's group
        
        print(f"Successfully loaded Karate Club network:")
        print(f"  - Nodes: {self.G.number_of_nodes()}")
        print(f"  - Edges: {self.G.number_of_edges()}")
        print(f"  - Ground truth communities: {len(set(self.ground_truth.values()))}")
    
    def detect_communities_girvan_newman(self, k=2):
        """Detect communities using Girvan-Newman algorithm"""
        print("Detecting communities with Girvan-Newman algorithm...")
        
        comp = nx.algorithms.community.girvan_newman(self.G)
        
        communities = []
        for i, communities_at_level in enumerate(comp):
            if len(communities_at_level) >= k or i >= 5:
                communities = list(communities_at_level)
                break
        
        partition = {}
        for comm_id, community in enumerate(communities):
            for node in community:
                partition[node] = comm_id
        
        return partition
    
    def detect_communities_spectral(self, k=2):
        """Detect communities using spectral clustering"""
        print("Detecting communities with Spectral Clustering...")
        
        nodes = list(self.G.nodes())
        adj_matrix = nx.to_numpy_array(self.G, nodelist=nodes)
        
        sc = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=42)
        labels = sc.fit_predict(adj_matrix)
        
        partition = {node: label for node, label in zip(nodes, labels)}
        return partition
    
    def evaluate_communities(self, detected_communities, algorithm_name):
        """Evaluate the detected communities against ground truth"""
        # Calculate NMI (Normalized Mutual Information)
        true_labels = [self.ground_truth[node] for node in sorted(self.ground_truth.keys())]
        detected_labels = [detected_communities[node] for node in sorted(detected_communities.keys())]
        
        nmi = normalized_mutual_info_score(true_labels, detected_labels)
        
        # Calculate modularity
        modularity = nx.algorithms.community.quality.modularity(
            self.G, 
            [set([node for node, comm in detected_communities.items() if comm == c]) 
             for c in set(detected_communities.values())]
        )
        
        print(f"\n{algorithm_name} Results:")
        print(f"  - Number of communities detected: {len(set(detected_communities.values()))}")
        print(f"  - Ground truth communities: {len(set(self.ground_truth.values()))}")
        print(f"  - Normalized Mutual Information (NMI): {nmi:.4f}")
        print(f"  - Modularity: {modularity:.4f}")
        
        return nmi, modularity
    
    def visualize_communities(self, detected_communities, title="Karate Club Communities"):
        """Visualize the network with detected communities"""
        plt.figure(figsize=(10, 8))
        
        # Create layout
        pos = nx.spring_layout(self.G, seed=42)
        
        # Get community colors
        communities = defaultdict(list)
        for node, comm_id in detected_communities.items():
            communities[comm_id].append(node)
        
        # Plot each community with different color
        colors = ['red', 'blue']
        for i, (comm_id, nodes) in enumerate(communities.items()):
            color = colors[i % len(colors)]
            nx.draw_networkx_nodes(
                self.G, pos, 
                nodelist=nodes,
                node_color=color,
                node_size=300,
                alpha=0.8,
                label=f'Community {comm_id}'
            )
        
        # Draw edges
        nx.draw_networkx_edges(self.G, pos, alpha=0.5)
        
        # Draw node labels
        labels = {node: str(node) for node in self.G.nodes()}
        nx.draw_networkx_labels(self.G, pos, labels, font_size=8)
        
        # Highlight the key figures (Mr. Hi and John A)
        nx.draw_networkx_nodes(self.G, pos, nodelist=[0], node_color='red', node_size=500, alpha=1.0)
        nx.draw_networkx_nodes(self.G, pos, nodelist=[33], node_color='blue', node_size=500, alpha=1.0)
        
        plt.title(title)
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def visualize_ground_truth(self):
        """Visualize the actual ground truth split"""
        plt.figure(figsize=(10, 8))
        
        pos = nx.spring_layout(self.G, seed=42)
        
        # Mr. Hi's group (red)
        mr_hi_group = [node for node, group in self.ground_truth.items() if group == 0]
        # John A's group (blue)  
        john_group = [node for node, group in self.ground_truth.items() if group == 1]
        
        nx.draw_networkx_nodes(self.G, pos, nodelist=mr_hi_group, node_color='red', 
                              node_size=300, alpha=0.8, label="Mr. Hi's Group")
        nx.draw_networkx_nodes(self.G, pos, nodelist=john_group, node_color='blue', 
                              node_size=300, alpha=0.8, label="John A's Group")
        
        # Highlight leaders
        nx.draw_networkx_nodes(self.G, pos, nodelist=[0], node_color='red', node_size=500, alpha=1.0)
        nx.draw_networkx_nodes(self.G, pos, nodelist=[33], node_color='blue', node_size=500, alpha=1.0)
        
        nx.draw_networkx_edges(self.G, pos, alpha=0.5)
        nx.draw_networkx_labels(self.G, pos, {node: str(node) for node in self.G.nodes()}, font_size=8)
        
        plt.title("Zachary Karate Club - Actual Split (Ground Truth)")
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline with both algorithms"""
        # Load data
        self.load_karate_data()
        
        print("\n" + "="*60)
        print("COMMUNITY DETECTION ANALYSIS - ZACHARY KARATE CLUB")
        print("="*60)
        
        # Show ground truth first
        self.visualize_ground_truth()
        
        results = {}
        
        # Test both algorithms
        algorithms = [
            ("Girvan-Newman", lambda: self.detect_communities_girvan_newman(k=2)),
            ("Spectral Clustering", lambda: self.detect_communities_spectral(k=2))
        ]
        
        for algo_name, algo_func in algorithms:
            try:
                print(f"\n{'='*40}")
                print(f"Running {algo_name}...")
                print('='*40)
                
                communities = algo_func()
                nmi, modularity = self.evaluate_communities(communities, algo_name)
                
                results[algo_name] = {
                    'communities': communities,
                    'nmi': nmi,
                    'modularity': modularity,
                    'num_communities': len(set(communities.values()))
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
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"{'Algorithm':<20} {'NMI':<10} {'Modularity':<12} {'Communities':<12}")
        print(f"{'-'*60}")
        
        for algo_name, result in results.items():
            print(f"{algo_name:<20} {result['nmi']:<10.4f} {result['modularity']:<12.4f} {result['num_communities']:<12}")
        
        # Find best algorithm
        if results:
            best_algo = max(results.items(), key=lambda x: x[1]['nmi'])
            print(f"\n🏆 Best Algorithm: {best_algo[0]} (NMI: {best_algo[1]['nmi']:.4f})")

# Run the analysis
if __name__ == "__main__":
    analyzer = KarateCommunityDetection()
    results = analyzer.run_complete_analysis()