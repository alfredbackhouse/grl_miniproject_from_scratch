import torch
import torch.nn as nn
from models.base_model import BaseModel
import torch.nn.functional as F

class VirtualNodeModel(BaseModel):
    def __init__(self, input_dim, hid_dim, n_classes, n_layers, dropout_ratio=0.3):
        """
        Virtual Node GCN Model with an additional virtual node.
        """
        super(VirtualNodeModel, self).__init__(input_dim, hid_dim, n_classes, n_layers, dropout_ratio)

        # Additional layer(s) for the virtual node
        self.virtual_node_mlp = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim)
        )

        self.attention_vector = nn.Parameter(torch.randn(hid_dim))

    def forward(self, X, A, virtual_node_feat=None):
        """
        Forward pass for the Virtual Node Model.

        Args:
            X: Node features (torch.Tensor) of shape [num_nodes, num_features]
            A: Edge index (torch.Tensor)
            virtual_node_feat: Virtual node feature vector (optional)

        Returns:
            logits: Node classification logits
            node_embeddings: Node embeddings from the last layer
        """
        if virtual_node_feat is None:
            virtual_node_feat = torch.zeros(1, self.hid_dim, device=X.device)  # Initialise virtual node

        # Apply GCN layers with virtual node interaction
        node_embeddings = X
        for layer in self.gcn_layers:
            node_embeddings = layer(node_embeddings, A)

            # Attention-based aggregation
            attention_weights = F.softmax(torch.matmul(node_embeddings, self.attention_vector), dim=0)  # [num_nodes]
            attention_weights = attention_weights.unsqueeze(1)  # [num_nodes, 1]
            aggregated = torch.sum(attention_weights * node_embeddings, dim=0, keepdim=True)  # [1, hid_dim]

            virtual_node_feat = self.virtual_node_mlp(virtual_node_feat + aggregated)

            # Broadcast virtual node back to node embeddings
            node_embeddings = node_embeddings + virtual_node_feat  # Match [num_nodes, hid_dim]
            node_embeddings = torch.relu(node_embeddings)
            node_embeddings = nn.functional.dropout(node_embeddings, p=self.dropout_ratio, training=self.training)

        # Classification layer
        logits = self.output_layer(node_embeddings)
        return logits, node_embeddings