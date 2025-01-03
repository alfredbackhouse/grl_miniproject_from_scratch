import torch
import torch.nn as nn
from models.base_model import BaseModel
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse, k_hop_subgraph

class BaselineGCNModel(BaseModel):
    def __init__(self, input_dim, hid_dim, n_classes, n_layers, dropout_ratio=0.3):
        """
        Virtual Node GCN Model with an additional virtual node.
        """
        super(BaselineGCNModel, self).__init__(input_dim, hid_dim, n_classes, n_layers, dropout_ratio)

    def forward(self, X, A):
        """
        Forward pass for the Baseline GCN Model.

        Args:
            X: Node features (torch.Tensor) of shape [num_nodes, num_features]
            A: Edge index (torch.Tensor)

        Returns:
            logits: Node classification logits
            node_embeddings: Node embeddings from the last layer
        """
        # rwse = self.rwse_proj(rwse)  # Shape: [num_nodes, input_dim]
        # X = X + rwse  # Combine RWSE with input features
        
        # edge_index = self.get_k_hop_edges(dense_to_sparse(A.cpu()), X.size(0), self.head_depth).to(device=A.device)
        node_embeddings = self.apply_gcn_layers(X, A)

        # Classification layer
        logits = self.output_layer(node_embeddings)
        return logits, node_embeddings
    
    def get_k_hop_edges(self, edge_index, num_nodes, k):
        """
        Compute the k-hop edges for the graph using PyTorch Geometric's k_hop_subgraph.

        Args:
            edge_index: Edge index (torch.Tensor) of shape [2, num_edges].
            num_nodes: Number of nodes in the graph.
            k: Number of hops (k-hop neighbourhood).

        Returns:
            k_hop_edge_index: Edge index including k-hop neighbourhoods.
        """
        # `k_hop_subgraph` generates the k-hop neighbourhood subgraph
        print(edge_index.shape)
        subset, k_hop_edge_index, _, _ = k_hop_subgraph(
            torch.arange(num_nodes, device=edge_index.device),  # Nodes to compute for
            k,
            edge_index,
            relabel_nodes=False
        )
        return k_hop_edge_index