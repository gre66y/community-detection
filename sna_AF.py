import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import requests
import zipfile
import io
from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import SpectralClustering

class FootballCommunityDetection:
    def __init__(self):
        self.G = None
        self.ground_truth = None
        
    def download_and_load_football_data(self):
        """Download and load the football dataset"""
        url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"
        
        try:
            # Download the zip file
            print("Downloading football dataset...")
            response = requests.get(url)
            response.raise_for_status()
            
            # Extract zip file
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Extract all files
                z.extractall('football_data')
                
                # Read the GML file
                self.G = nx.read_gml('football_data/football.gml')
                
                # Extract ground truth from node attributes
                self.ground_truth = {}
                for node, data in self.G.nodes(data=True):
                    if 'value' in data:
                        self.ground_truth[node] = data['value']
                
                print(f"Successfully loaded football network:")
                print(f"  - Nodes: {self.G.number_of_nodes()}")
                print(f"  - Edges: {self.G.number_of_edges()}")
                print(f"  - Ground truth communities: {len(set(self.ground_truth.values()))}")
                
        except Exception as e:
            print(f"Error downloading/loading data: {e}")
            self.load_fallback_data()
    
    def load_fallback_data(self):
        """Load fallback data if download fails"""
        print("Loading fallback football data...")
        try:
            # Try alternative download method
            self.G = nx.read_gml('http://www-personal.umich.edu/~mejn/netdata/football.gml')
            
            # Extract ground truth
            self.ground_truth = {}
            for node, data in self.G.nodes(data=True):
                if 'value' in data:
                    self.ground_truth[node] = data['value']
                    
        except:
            # Create synthetic football-like network
            self.G = nx.Graph()
            teams = list(range(115))
            conferences = {
                0: list(range(0, 10)),    # Conference 0
                1: list(range(10, 20)),   # Conference 1  
                2: list(range(20, 30)),   # Conference 2
                3: list(range(30, 40)),   # Conference 3
                4: list(range(40, 50)),   # Conference 4
                5: list(range(50, 60)),   # Conference 5
                6: list(range(60, 70)),   # Conference 6
                7: list(range(70, 80)),   # Conference 7
                8: list(range(80, 90)),   # Conference 8
                9: list(range(90, 100)),  # Conference 9
                10: list(range(100, 115)) # Conference 10
            }
            
            for team in teams:
                self.G.add_node(team)
                # Find which conference this team belongs to
                for conf_id, conf_teams in conferences.items():
                    if team in conf_teams:
                        self.ground_truth[team] = conf_id
                        break
            
            # Add edges within conferences (more dense)
            for conf_id, conf_teams in conferences.items():
                for i in range(len(conf_teams)):
                    for j in range(i+1, len(conf_teams)):
                        if np.random.random() < 0.6:  # 60% chance of connection within conference
                            self.G.add_edge(conf_teams[i], conf_teams[j])
            
            # Add some edges between conferences (less dense)
            for i in range(len(teams)):
                for j in range(i+1, len(teams)):
                    if self.ground_truth[teams[i]] != self.ground_truth[teams[j]]:
                        if np.random.random() < 0.1:  # 10% chance of connection between conferences
                            self.G.add_edge(teams[i], teams[j])
    
    def detect_communities_girvan_newman(self, k=12):
        """Detect communities using Girvan-Newman algorithm"""
        print("Detecting communities with Girvan-Newman algorithm...")
        
        # Girvan-Newman algorithm
        comp = nx.algorithms.community.girvan_newman(self.G)
        
        # Get the first k communities
        communities = []
        for i, communities_at_level in enumerate(comp):
            if len(communities_at_level) >= k or i >= 8:  # Stop condition
                communities = list(communities_at_level)
                print(f"Stopped at iteration {i} with {len(communities)} communities")
                break
        
        # Convert to partition format
        partition = {}
        for comm_id, community in enumerate(communities):
            for node in community:
                partition[node] = comm_id
        
        return partition
    
    def detect_communities_spectral(self, k=12):
        """Detect communities using spectral clustering"""
        print("Detecting communities with Spectral Clustering...")
        
        # Create adjacency matrix
        nodes = list(self.G.nodes())
        adj_matrix = nx.to_numpy_array(self.G, nodelist=nodes)
        
        # Apply spectral clustering
        sc = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=42)
        labels = sc.fit_predict(adj_matrix)
        
        # Convert to partition format
        partition = {node: label for node, label in zip(nodes, labels)}
        
        return partition
    
    def evaluate_communities(self, detected_communities, algorithm_name):
        """Evaluate the detected communities against ground truth"""
        if not self.ground_truth:
            print("No ground truth available for evaluation")
            return None, None
        
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
    
    def visualize_communities(self, detected_communities, title="Football Network Communities"):
        """Visualize the network with detected communities"""
        plt.figure(figsize=(12, 8))
        
        # Create layout
        pos = nx.spring_layout(self.G, seed=42)
        
        # Get community colors
        communities = defaultdict(list)
        for node, comm_id in detected_communities.items():
            communities[comm_id].append(node)
        
        # Plot each community with different color
        colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))
        for i, (comm_id, nodes) in enumerate(communities.items()):
            nx.draw_networkx_nodes(
                self.G, pos, 
                nodelist=nodes,
                node_color=[colors[i]],
                node_size=100,
                alpha=0.8,
                label=f'Community {comm_id}'
            )
        
        # Draw edges
        nx.draw_networkx_edges(self.G, pos, alpha=0.3)
        
        # Draw node labels (team names if available)
        if all(isinstance(node, str) for node in self.G.nodes()):
            labels = {node: node for node in self.G.nodes()}
            nx.draw_networkx_labels(self.G, pos, labels, font_size=6)
        
        plt.title(title)
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline with both algorithms"""
        # Load data
        self.download_and_load_football_data()
        
        print("\n" + "="*60)
        print("COMMUNITY DETECTION ANALYSIS - FOOTBALL NETWORK")
        print("="*60)
        print("Models: Girvan-Newman & Spectral Clustering")
        
        results = {}
        
        # Test both algorithms
        algorithms = [
            ("Girvan-Newman", lambda: self.detect_communities_girvan_newman(k=12)),
            ("Spectral Clustering", lambda: self.detect_communities_spectral(k=12))
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
            if result['nmi'] is not None:
                print(f"{algo_name:<20} {result['nmi']:<10.4f} {result['modularity']:<12.4f} {result['num_communities']:<12}")
        
        # Find best algorithm
        if results:
            best_algo = max(results.items(), key=lambda x: x[1]['nmi'])
            print(f"\n🏆 Best Algorithm: {best_algo[0]} (NMI: {best_algo[1]['nmi']:.4f})")

# Run the analysis
if __name__ == "__main__":
    analyzer = FootballCommunityDetection()
    results = analyzer.run_complete_analysis()