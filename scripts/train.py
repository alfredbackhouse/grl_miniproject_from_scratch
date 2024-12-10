import torch
import yaml
from models.virtual_node import VirtualNodeModel
from models.baseline_gcn import BaselineGCNModel
import torch_geometric
from torch_geometric.loader import DataLoader
import yaml
from data.utils import load_planetoid_dataset, hierarchical_clustering

# Load Planetoid dataset
dataset = load_planetoid_dataset(name="Cora", root="./data/Planetoid")
data = dataset[0]  # Planetoid datasets typically have a single graph
hiearchical_data = hierarchical_clustering(data, num_clusters_per_level=[32, 8, 1], num_levels=3)
print("HSG:", hiearchical_data)
data = hiearchical_data

with open("experiments/config.yaml", "r") as f:
    config = yaml.safe_load(f)
# Initialise model from config
model_config = config["model"]

# Define model, loss, and optimiser
model = BaselineGCNModel(
    input_dim=data.num_node_features,
    hid_dim=model_config["hid_dim"],
    n_classes=dataset.num_classes,
    n_layers=model_config["n_layers"], 
    dropout_ratio=model_config["dropout_ratio"]
)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
def train():
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        logits, embeddings = model(data.x, data.edge_index)
        
        # Compute training loss
        train_loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        optimizer.step()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(data.x, data.edge_index)
            val_loss = criterion(val_logits[data.val_mask], data.y[data.val_mask])
            
            # Compute validation accuracy
            val_preds = val_logits[data.val_mask].argmax(dim=1)
            val_labels = data.y[data.val_mask]
            val_accuracy = (val_preds == val_labels).sum().item() / data.val_mask.sum().item()
        
        model.train()  # Switch back to training mode
        
        # Print metrics
        print(f"Epoch {epoch+1}: Train Loss = {train_loss.item():.4f}, Val Loss = {val_loss.item():.4f}, Val Accuracy = {val_accuracy:.4f}")

if __name__ == "__main__":
    train()