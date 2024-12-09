import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class BaseModel(nn.Module):
    def __init__(self, input_dim: int, hid_dim: int, n_classes: int, n_layers: int, dropout_ratio: float = 0.3):
        """
        Base model class for GCN-based architectures.

        Args:
            input_dim: Input feature dimension
            hid_dim: Hidden feature dimension
            n_classes: Number of target classes
            n_layers: Number of GCN layers
            dropout_ratio: Dropout ratio for regularisation
        """
        super(BaseModel, self).__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.dropout_ratio = dropout_ratio

        # Shared layer initialisation logic
        self.gcn_layers = nn.ModuleList()
        if n_layers > 0:
            self.gcn_layers.append(GCNConv(input_dim, hid_dim))
            for _ in range(n_layers - 1):
                self.gcn_layers.append(GCNConv(hid_dim, hid_dim))

        # Output layer
        self.output_layer = nn.Linear(hid_dim, n_classes)

    def forward(self, X, A):
        """
        Forward pass for the base GCN model. Should be overridden if needed.
        """
        raise NotImplementedError("Forward method should be implemented in subclasses.")

    def apply_gcn_layers(self, X, A):
        """
        Pass node features through GCN layers with ReLU and Dropout.

        Args:
            X: Node features (torch.Tensor)
            A: Edge index (torch.Tensor)

        Returns:
            Node embeddings after all GCN layers.
        """
        for layer in self.gcn_layers:
            X = layer(X, A)
            X = torch.relu(X)
            X = nn.functional.dropout(X, p=self.dropout_ratio, training=self.training)
        return X