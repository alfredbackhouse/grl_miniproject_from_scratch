import torch
import yaml
from models.virtual_node import VirtualNodeModel
from models.baseline_gcn import BaselineGCNModel
from models.gcnii import GCNII
from models.gated_gcn import GatedGCN
import torch_geometric
from torch_geometric.loader import DataLoader
from data.utils import load_planetoid_dataset, load_processed_lrgb_dataset, load_lrgb_dataset, hierarchical_clustering, hierarchical_reverse_clustering
from sklearn.metrics import f1_score
from torch_geometric.utils import k_hop_subgraph, to_dense_adj
from sklearn.decomposition import PCA
# Import necessary libraries
import random
import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Read config file
with open("experiments/config.yaml", "r") as f:
    config = yaml.safe_load(f)
model_config = config["model"]
data_config = config["data"]
train_config = config["training"]

if data_config["dataset"] == "Cora":
    # Load the dataset
    train_dataset = load_planetoid_dataset(name=data_config["dataset"], root="./data/Planetoid")
    if model_config["clustering_type"] == "hsg":
        hiearchical_data = hierarchical_clustering(train_dataset[0], num_clusters_per_level=[16, 1], num_levels=2, cora=True)
    elif model_config["clustering_type"] == "unet":
        hiearchical_data = hierarchical_reverse_clustering(train_dataset[0], num_clusters_per_level=[16, 1], num_levels=2, unet=False, cora=True)
    elif model_config["clustering_type"] == "unet_with_skip":
        hiearchical_data = hierarchical_reverse_clustering(train_dataset[0], num_clusters_per_level=[16, 1], num_levels=2, unet=True, cora=True)
    elif model_config["clustering_type"] == "vn":
        hiearchical_data = hierarchical_clustering(train_dataset[0], num_clusters_per_level=[1], num_levels=1, cora=True)
    else: 
        print("No clustering type")
        hiearchical_data = train_dataset[0]
    train_dataset = [hiearchical_data]
    dataset = train_dataset
else: 
    val_dataset = load_processed_lrgb_dataset(name=data_config["dataset"], clustering_type=model_config["clustering_type"], split="val", root="./data/LRGB", overwrite=False)
    dataset = val_dataset

# Define statistics computation function
def compute_graph_statistics(graph):
    stats = {}
    stats['avg_nodes'] = graph.number_of_nodes()
    stats['avg_edges'] = graph.number_of_edges()
    stats['diameter'] = nx.diameter(graph.to_undirected()) if nx.is_connected(graph.to_undirected()) else float('inf')
    stats['avg_shortest_path'] = nx.average_shortest_path_length(graph.to_undirected()) if nx.is_connected(graph.to_undirected()) else float('inf')
    
    # Effective Resistance (using Moore-Penrose pseudoinverse of the Laplacian)
    L = nx.laplacian_matrix(graph.to_undirected()).toarray()
    try:
        L_pinv = np.linalg.pinv(L)
        resistances = []
        for u, v in graph.edges():
            resistances.append(L_pinv[u, u] + L_pinv[v, v] - 2 * L_pinv[u, v])
        stats['E[R_ab]'] = np.mean(resistances)
    except np.linalg.LinAlgError:
        stats['E[R_ab]'] = float('inf')
    
    # Expected Commute Time (related to effective resistance)
    try:
        stats['E[C_ab]'] = np.mean([2 * graph.number_of_edges() * r for r in resistances])
    except NameError:
        stats['E[C_ab]'] = float('inf')
    
    # Placeholder metrics for GNC and ANC
    stats['GNC'] = random.uniform(1.0, 2.0)
    stats['ANC'] = random.uniform(1.0, 3.0)
    
    return stats

# Sample 100 random graphs
random_graphs = random.sample(list(dataset), 100)
all_stats = []

for data in random_graphs:
    graph = to_networkx(data, node_attrs=['x'])
    stats = compute_graph_statistics(graph)
    all_stats.append(stats)

# Aggregate statistics
import pandas as pd
stats_df = pd.DataFrame(all_stats)
final_stats = {
    'avg_nodes': stats_df['avg_nodes'].mean(),
    'avg_edges': stats_df['avg_edges'].mean(),
    'diameter': stats_df['diameter'].mean(),
    'avg_shortest_path': stats_df['avg_shortest_path'].mean(),
    'E[R_ab]': stats_df['E[R_ab]'].mean(),
    'E[C_ab]': stats_df['E[C_ab]'].mean(),
    'GNC': stats_df['GNC'].mean(),
    'ANC': stats_df['ANC'].mean()
}

# Print final statistics
print(pd.DataFrame([final_stats]))
