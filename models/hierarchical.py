import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel
from sklearn.cluster import KMeans
from torch_geometric.utils import to_dense_adj

class HierachicalModel(BaseModel):
    def __init__(self, input_dim, hid_dim, n_classes, n_layers, dropout_ratio=0.3, num_clusters=10):
        """
        Virtual Node GCN Model with Clustering.
        """
        super(HierachicalModel, self).__init__(input_dim, hid_dim, n_classes, n_layers, dropout_ratio)

        # Define the virtual node MLP
        self.virtual_node_mlp = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim)
        )

        # Attention vector for node importance
        self.attention_vector = nn.Parameter(torch.randn(hid_dim))

        # Clustering parameters
        self.num_clusters = num_clusters

    def perform_clustering(self, node_embeddings):
        """
        Perform k-means clustering on node embeddings.
        
        Args:
            node_embeddings (torch.Tensor): Node embeddings of shape [num_nodes, hid_dim].

        Returns:
            cluster_assignments (torch.Tensor): Tensor of shape [num_nodes] with cluster indices.
        """
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        cluster_assignments = kmeans.fit_predict(node_embeddings.detach().cpu().numpy())
        return torch.tensor(cluster_assignments, device=node_embeddings.device)

    def aggregate_clusters(self, node_embeddings, cluster_assignments):
        """
        Aggregate node features into cluster-level features.

        Args:
            node_embeddings (torch.Tensor): Node embeddings of shape [num_nodes, hid_dim].
            cluster_assignments (torch.Tensor): Cluster assignments of shape [num_nodes].

        Returns:
            cluster_features (torch.Tensor): Aggregated cluster features of shape [num_clusters, hid_dim].
        """
        cluster_features = torch.zeros(self.num_clusters, node_embeddings.size(1), device=node_embeddings.device)
        for cluster_id in range(self.num_clusters):
            cluster_mask = (cluster_assignments == cluster_id)
            cluster_features[cluster_id] = node_embeddings[cluster_mask].mean(dim=0)
        return cluster_features

    def forward(self, X, A):
        """
        Forward pass for the Virtual Node Model.

        Args:
            X: Node features (torch.Tensor) of shape [num_nodes, num_features]
            A: Edge index (torch.Tensor)

        Returns:
            logits: Node classification logits
            node_embeddings: Node embeddings from the last layer
        """
        # Step 1: Apply GCN layers to obtain node embeddings
        node_embeddings = self.apply_gcn_layers(X, A)

        # Step 2: Perform clustering on node embeddings
        cluster_assignments = self.perform_clustering(node_embeddings)

        # Step 3: Aggregate cluster-level features
        cluster_features = self.aggregate_clusters(node_embeddings, cluster_assignments)

        # Step 4: Attention-based aggregation for virtual node
        attention_weights = F.softmax(torch.matmul(node_embeddings, self.attention_vector), dim=0)  # [num_nodes]
        attention_weights = attention_weights.unsqueeze(1)  # [num_nodes, 1]
        aggregated = torch.sum(attention_weights * node_embeddings, dim=0, keepdim=True)  # [1, hid_dim]

        virtual_node_feat = self.virtual_node_mlp(aggregated)

        # Step 5: Broadcast virtual node back to node embeddings
        node_embeddings = node_embeddings + virtual_node_feat  # Match [num_nodes, hid_dim]

        # Step 6: Classification layer
        logits = self.output_layer(node_embeddings)
        return logits, node_embeddings, cluster_features