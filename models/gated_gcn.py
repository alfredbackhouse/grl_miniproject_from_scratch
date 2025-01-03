import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from models.base_model import BaseModel

class GatedGCNLayer(MessagePassing):
    def __init__(self, input_dim, output_dim, dropout_ratio=0.3):
        super(GatedGCNLayer, self).__init__(aggr='add')  # Aggregation is 'add'
        self.node_linear = nn.Linear(input_dim, output_dim)
        self.edge_linear = nn.Linear(input_dim, output_dim)
        self.gate_linear = nn.Linear(2 * output_dim, 1)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass for the Gated GCN layer.
        
        Args:
            x: Node features of shape [num_nodes, input_dim].
            edge_index: Edge indices of shape [2, num_edges].
            edge_attr: Edge features of shape [num_edges, input_dim].
        
        Returns:
            Node features of shape [num_nodes, output_dim].
        """
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        edge_attr = self.edge_linear(edge_attr) if edge_attr is not None else None

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        """
        Compute the message for each edge.
        
        Args:
            x_i: Source node features [num_edges, output_dim].
            x_j: Target node features [num_edges, output_dim].
            edge_attr: Edge features [num_edges, output_dim].
        
        Returns:
            Messages with gating applied [num_edges, output_dim].
        """
        edge_input = torch.cat([x_i, x_j], dim=-1)
        gate = torch.sigmoid(self.gate_linear(edge_input))  # Compute the gate
        if edge_attr is not None:
            edge_features = x_j + edge_attr
        else:
            edge_features = x_j
        return gate * edge_features  # Apply gate to message

    def update(self, aggr_out):
        """
        Update the node embeddings after aggregation.
        """
        return self.dropout(aggr_out)

# Gated GCN Model inheriting from BaseModel
class GatedGCN(BaseModel):
    def __init__(self, input_dim: int, hid_dim: int, n_classes: int, n_layers: int, dropout_ratio: float = 0.3):
        """
        Gated GCN Model that inherits from BaseModel.
        """
        super(GatedGCN, self).__init__(input_dim, hid_dim, n_classes, n_layers, dropout_ratio)
        self.gated_layers = nn.ModuleList()
        self.edge_features = True  # Assume edge features are provided

        # Initialize gated GCN layers
        self.gated_layers.append(GatedGCNLayer(input_dim, hid_dim, dropout_ratio))
        for _ in range(n_layers - 1):
            self.gated_layers.append(GatedGCNLayer(hid_dim, hid_dim, dropout_ratio))

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass for the Gated GCN Model.
        
        Args:
            x: Node features of shape [num_nodes, input_dim].
            edge_index: Edge indices of shape [2, num_edges].
            edge_attr: Edge features of shape [num_edges, input_dim].
        
        Returns:
            logits (torch.Tensor): Predictions for each node of shape [num_nodes, n_classes].
            embeddings (torch.Tensor): Final node embeddings of shape [num_nodes, hid_dim].
        """
        for layer in self.gated_layers:
            x = layer(x, edge_index, edge_attr)

        # Pass through the output layer for classification
        logits = self.output_layer(x)
        return logits, x