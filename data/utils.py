import torch
import torch_geometric
import os
from torch_geometric.datasets import Planetoid, LRGBDataset
from torch_geometric.transforms import GCNNorm
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from sklearn.cluster import KMeans
from torch_geometric.utils import from_networkx
import numpy as np 

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

def load_processed_planetoid_dataset(name: str, root: str = "./", transform=GCNNorm(), clustering_type=None) -> torch_geometric.data.Dataset:
    dataset = load_planetoid_dataset(name=name, root=root, transform=transform)
    if clustering_type == 'hierarchical':
        data = dataset[0]
        hierarchical_data = hierarchical_reverse_clustering(data, num_clusters_per_level=[16, 1], num_levels=2, unet=True)
        return hierarchical_data
    
def load_lrgb_dataset(name: str, split="train", root: str = "./data/LRGB", transform=GCNNorm()) -> torch_geometric.data.Dataset:
    """
    Loads the LRGB dataset.

    Args:
        name (str): Name of the dataset ('Lung', 'Retina', 'Gland').
        root (str): Root directory for the dataset.
        transform (callable): Transformations to apply to the dataset.

    Returns:
        torch_geometric.data.Dataset: The loaded dataset.
    """
    dataset = LRGBDataset(root=root, name=name, transform=transform, split=split)
    print(f"Loaded {name} dataset with {len(dataset)} graphs.")
    print(dataset)
    print(vars(dataset))
    print(dataset[0])  # Inspect dataset structure
    return dataset

def load_processed_lrgb_dataset(name: str, clustering_type:str, split="train", root: str = "./data/LRGB", overwrite:bool=False) -> torch_geometric.data.Dataset:
    """
    Process and save an LRGB dataset with a specific hierarchical clustering.

    Args:
        clustering_type (str): Type of clustering ('standard', 'reverse', etc.).
        clustering_function (callable): The clustering function to use.
        root (str): Directory to save the processed graphs.
        overwrite (bool): If True, overwrite existing processed graphs.

    Returns:
        Dataset: The processed dataset.
    """

    # Load the original dataset
    dataset = load_lrgb_dataset(name=name, root=root, split=split)

    if clustering_type == "none":
        return dataset

    # Define the save directory based on clustering type
    dataset_root = os.path.join(root, name)
    save_dir = os.path.join(dataset_root, f"processed_{split}_{clustering_type}")
    os.makedirs(save_dir, exist_ok=True)

    processed_graphs = []

    # Check if clustering is already done
    for i, graph in enumerate(dataset):
        save_path = os.path.join(save_dir, f"graph_{i}.pt")
        if not os.path.exists(save_path) or overwrite:
            print(f"Processing graph {i + 1}/{len(dataset)}...")
            if clustering_type == "unet":
                clustered_graph = hierarchical_reverse_clustering(graph, num_clusters_per_level=[16, 1], num_levels=2, unet=False)
            elif clustering_type == "unet_with_skip":
                clustered_graph = hierarchical_reverse_clustering(graph, num_clusters_per_level=[16, 1], num_levels=2, unet=True)
            elif clustering_type == "hsg":
                clustered_graph = hierarchical_clustering(graph, num_clusters_per_level=[16, 1], num_levels=2)
            elif clustering_type == "vn":
                clustered_graph = hierarchical_clustering(graph, num_clusters_per_level=[1], num_levels=1)
            torch.save(clustered_graph, save_path)
        else:
            print(f"Loading preprocessed graph {i + 1}/{len(dataset)}...")
            clustered_graph = torch.load(save_path)

        processed_graphs.append(clustered_graph)
        # break

    # Return the processed graphs as a new dataset-like object
    return processed_graphs

def hierarchical_clustering(data, num_clusters_per_level, num_levels, cora=False):
    """
    Perform hierarchical clustering on a graph and compute a single hierarchical graph
    with edges from nodes in the current layer to their corresponding cluster summary node,
    while retaining the train/val/test masks for the original nodes only.

    Args:
        data (torch_geometric.data.Data): Input graph with `x` (features), `edge_index`, and masks.
        num_clusters_per_level (list): Number of clusters at each level (one per level).
        num_levels (int): Number of levels in the hierarchy.

    Returns:
        hierarchical_graph (torch_geometric.data.Data): Graph with original nodes,
            cluster summary nodes, and connections between them, retaining the original masks.
    """
    if num_levels == 0: 
        return data
    current_features = data.x
    current_edge_index = data.edge_index
    all_node_features = [data.x]  # Store all node features (original + cluster nodes)
    all_edge_indices = [data.edge_index]  # Store all edges (original + cluster edges)
    current_node_offset = 0  # Tracks the offset of nodes in the most recent layer

    for level in range(num_levels):
        num_clusters = num_clusters_per_level[level]

        # Cluster nodes in the current layer
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_assignments = kmeans.fit_predict(current_features.cpu().detach().numpy())
        cluster_assignments = torch.tensor(cluster_assignments, device=current_features.device)

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
            src_cluster = cluster_assignments[edge[0] - current_node_offset]
            dst_cluster = cluster_assignments[edge[1] - current_node_offset]
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

    # Mark the final graph summary node
    is_graph_summary = torch.zeros(combined_features.size(0), dtype=torch.bool, device=combined_features.device)
    is_graph_summary[-1] = True  # Mark the last virtual node as the graph summary node

    # Extend masks to include only the original nodes
    y = torch.cat([data.y, torch.zeros(combined_features.size(0) - data.x.size(0), dtype=torch.long)])
    if cora: 
        train_mask = torch.cat([data.train_mask, torch.zeros(combined_features.size(0) - data.x.size(0), dtype=torch.bool)])
        val_mask = torch.cat([data.val_mask, torch.zeros(combined_features.size(0) - data.x.size(0), dtype=torch.bool)])
        test_mask = torch.cat([data.test_mask, torch.zeros(combined_features.size(0) - data.x.size(0), dtype=torch.bool)])

    # Return the final graph with retained masks
    final_graph = Data(x=combined_features, edge_index=combined_edge_index)
    final_graph.y = y
    final_graph.is_graph_summary = is_graph_summary
    if cora: 
        final_graph.train_mask = train_mask
        final_graph.val_mask = val_mask
        final_graph.test_mask = test_mask

    return final_graph

def hierarchical_reverse_clustering(data, num_clusters_per_level, num_levels, unet=False, cora=False):
    """
    Create a hierarchical UNet-like structure with a bottleneck layer.

    Args:
        data (torch_geometric.data.Data): Input graph with `x` (features) and `edge_index`.
        num_clusters_per_level (list): Number of clusters at each level (one per level).
        num_levels (int): Number of levels in the hierarchy.

    Returns:
        reverse_graph (torch_geometric.data.Data): Graph with hierarchical nodes (forward and reverse),
            including a bottleneck layer, and connections back to the original nodes.
    """
    if num_levels == 0:
        return data
    current_features = data.x
    current_edge_index = data.edge_index
    all_node_features = [data.x]  # Store all node features (original + hierarchical nodes)
    all_edge_indices = [data.edge_index]  # Store all edges (original + hierarchical edges)
    current_node_offset = 0  # Tracks the offset of nodes in the most recent layer
    reverse_node_features = []  # Store reverse nodes
    reverse_edges = []  # Edges for reverse connections
    level_offsets = [0]  # Offsets for each layer

    original_to_clusters = {}  # Track mappings from original nodes to clusters
    forward_intra_layer_edges = []  # Store intra-layer edges for reverse layers
    forward_inter_layer_edges = []  # Store inter-layer edges for reverse layers

    # Forward pass: Build hierarchical layers
    for level in range(num_levels):
        num_clusters = num_clusters_per_level[level]

        # Cluster nodes in the current layer
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_assignments = kmeans.fit_predict(current_features.cpu().detach().numpy())
        cluster_assignments = torch.tensor(cluster_assignments, device=current_features.device)

        # Aggregate features for each cluster
        cluster_features = torch.zeros(num_clusters, current_features.size(1), device=current_features.device)
        for cluster_id in range(num_clusters):
            cluster_mask = (cluster_assignments == cluster_id)
            cluster_features[cluster_id] = current_features[cluster_mask].mean(dim=0)

        # Create unidirectional edges from current layer nodes to their cluster summary nodes
        virtual_node_offset = sum(f.size(0) for f in all_node_features)  # Offset for virtual node indices
        virtual_edges = []
        for node_idx in range(current_features.size(0)):
            cluster_id = cluster_assignments[node_idx]
            virtual_node_idx = virtual_node_offset + cluster_id
            virtual_edges.append((torch.tensor(current_node_offset, dtype=torch.int32) + node_idx, virtual_node_idx))  # Edge to virtual node

            # Update original-to-cluster mapping so that at the end of the reverse pass, we can map back to the correct original nodes
            if level == 0:  # Map from original nodes to the first layer
                original_to_clusters[node_idx] = cluster_id + virtual_node_offset

        # Create edges between cluster summary nodes
        cluster_edges = set()
        for edge in current_edge_index.t().tolist():
            src_cluster = cluster_assignments[edge[0] - current_node_offset]
            dst_cluster = cluster_assignments[edge[1] - current_node_offset]
            if src_cluster != dst_cluster:
                src_virtual = virtual_node_offset + src_cluster
                dst_virtual = virtual_node_offset + dst_cluster
                cluster_edges.add((src_virtual, dst_virtual))
                cluster_edges.add((dst_virtual, src_virtual))

        # Save inter-layer edges for reverse pass
        forward_inter_layer_edges.append(torch.tensor(virtual_edges, dtype=torch.long).t())

        # Save intra-layer edges for reverse pass
        forward_intra_layer_edges.append(torch.tensor(list(cluster_edges), dtype=torch.long).t())
        level_offsets.append(virtual_node_offset) # Save offset for this layer

        # Update node features and edge indices
        all_node_features.append(cluster_features) # Add this level of nodes
        all_edge_indices.append(torch.tensor(virtual_edges, dtype=torch.long).t()) # Add edges to virtual nodes from prev level
        all_edge_indices.append(forward_intra_layer_edges[-1])  # Add intra-layer edges

        # Update features and edge_index for the next layer
        current_features = cluster_features
        current_edge_index = torch.tensor(list(cluster_edges), dtype=torch.long).t()
        current_node_offset = virtual_node_offset  # Update offset to reflect the current layer

        # Mark the bottleneck layer
        if level == num_levels - 1:
            bottleneck_offset = virtual_node_offset
            bottleneck_features = cluster_features

    prev_reverse_node_offset = bottleneck_offset  # Start from the bottleneck layer
    # Reverse pass: Copy hierarchical nodes (including inter and intra-layer edges)
    for level in range(num_levels - 1, -1, -1):
        num_clusters = num_clusters_per_level[level]
        reverse_node_offset = sum(f.size(0) for f in all_node_features + reverse_node_features)  # Offset for reverse nodes

        reverse_features = all_node_features[level].clone()  # Copy hierarchical nodes from the forward pass

        extra_edges = []
        if unet and level > 0:
            # Add skip connections
            for old_idx in range(reverse_features.size(0)):
                new_idx = old_idx + reverse_node_offset
                extra_edges.append((old_idx + level_offsets[level], new_idx))
        reverse_edges.append(torch.tensor(extra_edges, dtype=torch.int32).t())

        # Add intra-layer edges for reverse nodes
        if level > 0:
            reverse_intra_layer_edges = forward_intra_layer_edges[level-1].clone()
            reverse_intra_layer_edges += reverse_node_offset - level_offsets[level]
            reverse_edges.append(reverse_intra_layer_edges)

        reverse_inter_layer_edges = torch.flip(forward_inter_layer_edges[level].clone(), dims=[0])
        reverse_inter_layer_edges[0] += prev_reverse_node_offset - level_offsets[level+1]
        if level > 0: # if level = 0 we're gonna connect back to the original nodes
            reverse_inter_layer_edges[1] += reverse_node_offset - level_offsets[level]
        reverse_edges.append(reverse_inter_layer_edges)

        if level > 0: 
            reverse_node_features.append(reverse_features)

        prev_reverse_node_offset = reverse_node_offset

    # Combine forward and reverse node features and edges (including bottleneck)
    combined_features = torch.cat(all_node_features + reverse_node_features, dim=0)
    combined_edge_index = torch.cat(all_edge_indices + reverse_edges, dim=1)

    # Mark the final graph summary node
    is_graph_summary = torch.zeros(combined_features.size(0), dtype=torch.bool, device=combined_features.device)
    is_graph_summary[bottleneck_offset] = True  # Mark the last virtual node as the graph summary node

    # Extend masks to include only the original nodes
    y = torch.cat([data.y, torch.zeros(combined_features.size(0) - data.x.size(0), dtype=torch.long)])
    if cora: 
        train_mask = torch.cat([data.train_mask, torch.zeros(combined_features.size(0) - data.x.size(0), dtype=torch.bool)])
        val_mask = torch.cat([data.val_mask, torch.zeros(combined_features.size(0) - data.x.size(0), dtype=torch.bool)])
        test_mask = torch.cat([data.test_mask, torch.zeros(combined_features.size(0) - data.x.size(0), dtype=torch.bool)])

    # Create the final hierarchical UNet graph
    reverse_graph = Data(x=combined_features, edge_index=combined_edge_index)
    reverse_graph.y = y
    reverse_graph.is_graph_summary = is_graph_summary
    if cora: 
        reverse_graph.train_mask = train_mask
        reverse_graph.val_mask = val_mask
        reverse_graph.test_mask = test_mask
    return reverse_graph