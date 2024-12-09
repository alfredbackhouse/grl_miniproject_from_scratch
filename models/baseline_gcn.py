import torch
import torch.nn as nn
from models.base_model import BaseModel
import torch.nn.functional as F

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
        
        node_embeddings = self.apply_gcn_layers(X, A)

        # Classification layer
        logits = self.output_layer(node_embeddings)
        return logits, node_embeddings