import torch
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import GCNNorm
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from sklearn.cluster import KMeans


def load_planetoid_dataset(name: str, root: str = "./", transform=GCNNorm()) -> torch_geometric.data.Dataset:
    """
    Loads the Planetoid dataset.

    Args:
        name (str): Name of the dataset ('Cora', 'CiteSeer', 'PubMed').
        root (str): Root directory for the dataset.
        transform (callable): Transformations to apply to the dataset.

    Returns:
        torch_geometric.data.Dataset: The loaded dataset.
    """
    dataset = Planetoid(root=root, name=name, split="public", transform=transform)
    print(f"Loaded {name} dataset with {len(dataset)} graphs.")
    print(dataset.data)  # Inspect dataset structure
    return dataset


def hierarchical_clustering(data, num_clusters_per_level, num_levels):
    """
    Perform hierarchical clustering on a graph and compute a single hierarchical graph
    with edges from nodes in the current layer to their corresponding cluster summary node.

    Args:
        data (torch_geometric.data.Data): Input graph with `x` (features) and `edge_index`.
        num_clusters_per_level (list): Number of clusters at each level (one per level).
        num_levels (int): Number of levels in the hierarchy.

    Returns:
        hierarchical_graph (torch_geometric.data.Data): Graph with original nodes,
            cluster summary nodes, and connections between them.
    """
    current_features = data.x
    current_edge_index = data.edge_index
    all_node_features = [data.x]  # Store all node features (original + cluster nodes)
    all_edge_indices = [data.edge_index]  # Store all edges (original + cluster edges)
    current_node_offset = 0  # Tracks the offset of nodes in the most recent layer

    for level in range(num_levels):
        print("Current features:", current_features)
        print("Current edge index:", current_edge_index)
        num_clusters = num_clusters_per_level[level]

        # Cluster nodes in the current layer
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_assignments = kmeans.fit_predict(current_features.cpu().detach().numpy())
        cluster_assignments = torch.tensor(cluster_assignments, device=current_features.device)

        print("Cluster assignments:", cluster_assignments)

        # Aggregate features for each cluster
        cluster_features = torch.zeros(num_clusters, current_features.size(1), device=current_features.device)
        for cluster_id in range(num_clusters):
            cluster_mask = (cluster_assignments == cluster_id)
            cluster_features[cluster_id] = current_features[cluster_mask].mean(dim=0)

        # Create edges from current layer nodes to their cluster summary node
        virtual_node_offset = sum(f.size(0) for f in all_node_features)  # Offset for virtual node indices
        virtual_edges = []
        for node_idx in range(current_features.size(0)):
            cluster_id = cluster_assignments[node_idx]
            virtual_node_idx = virtual_node_offset + cluster_id
            virtual_edges.append([current_node_offset + node_idx, virtual_node_idx])  # Edge to virtual node
            virtual_edges.append([virtual_node_idx, current_node_offset + node_idx])  # Reverse edge

        # Create edges between cluster summary nodes
        cluster_edges = set()
        for edge in current_edge_index.t().tolist():
            src_cluster = cluster_assignments[edge[0] - virtual_node_offset]
            dst_cluster = cluster_assignments[edge[1] - virtual_node_offset]
            if src_cluster != dst_cluster:
                src_virtual = virtual_node_offset + src_cluster
                dst_virtual = virtual_node_offset + dst_cluster
                cluster_edges.add((src_virtual, dst_virtual))
                cluster_edges.add((dst_virtual, src_virtual))

        # Update node features and edge indices
        all_node_features.append(cluster_features)
        all_edge_indices.append(torch.tensor(virtual_edges, dtype=torch.long).t())
        all_edge_indices.append(torch.tensor(list(cluster_edges), dtype=torch.long).t())

        # Update features and edge_index for the next layer
        current_features = cluster_features
        current_edge_index = torch.tensor(list(cluster_edges), dtype=torch.long).t()
        current_node_offset = virtual_node_offset  # Update offset to reflect the current layer

    # Combine all node features and edges into a single graph
    combined_features = torch.cat(all_node_features, dim=0)
    combined_edge_index = torch.cat(all_edge_indices, dim=1)

    final_graph = Data(x=combined_features, edge_index=combined_edge_index)
    return final_graph