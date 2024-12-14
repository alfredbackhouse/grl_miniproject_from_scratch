import torch
import yaml
from models.virtual_node import VirtualNodeModel
from models.baseline_gcn import BaselineGCNModel
import torch_geometric
from torch_geometric.loader import DataLoader
from data.utils import load_planetoid_dataset, load_processed_lrgb_dataset
from sklearn.metrics import f1_score


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Read config file
with open("experiments/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load the dataset (train split)
train_dataset = load_processed_lrgb_dataset(name="PascalVOC-SP", clustering_type="unet", split="train", root="./data/LRGB", overwrite=False)
print(train_dataset[0])
# Create a DataLoader for batching
train_config = config["training"]
train_loader = DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True)

# Initialise model from config
model_config = config["model"]
model = BaselineGCNModel(
    input_dim=train_dataset[0].x.size(1),
    hid_dim=model_config["hid_dim"],
    n_classes=21,
    n_layers=model_config["n_layers"],
    dropout_ratio=model_config["dropout_ratio"]
).to(device)  # Move model to GPU
model.reset_parameters()

# Collect all labels from the dataset
all_labels = torch.cat([data.y for data in train_dataset], dim=0)

# Count occurrences of each class
num_classes = int(all_labels.max()) + 1  # Assuming labels are 0-indexed
class_counts = torch.bincount(all_labels, minlength=num_classes)
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum()  # Normalise

criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)  # Move criterion to GPU
optimizer = torch.optim.Adam(model.parameters(), lr=train_config["learning_rate"], weight_decay=train_config["weight_decay"])

# Training loop for node classification
def train():
    for epoch in range(train_config["epochs"]):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)  # Move batch to GPU
            optimizer.zero_grad()
            
            # Forward pass
            logits, embeddings = model(batch.x, batch.edge_index)
            
            # Compute loss for training nodes
            train_loss = criterion(logits, batch.y)
            train_loss.backward()
            optimizer.step()
            
            total_loss += train_loss.item()
        

        # Validation phase
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in train_loader:  # Use validation DataLoader here if available
                batch = batch.to(device)  # Move batch to GPU
                val_logits, _ = model(batch.x, batch.edge_index)
                batch_val_loss = criterion(val_logits, batch.y)
                val_loss += batch_val_loss.item()
                
                # Collect predictions and labels
                val_preds = val_logits.argmax(dim=1).cpu().numpy()
                val_labels = batch.y.cpu().numpy()
                all_preds.extend(val_preds)
                all_labels.extend(val_labels)

        # Calculate F1-score
        val_f1 = f1_score(all_labels, all_preds, average="macro")  # Weighted F1-score

        # Print metrics
        print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}, Val Loss = {val_loss:.4f}, Val F1-Score = {val_f1:.4f}")

if __name__ == "__main__":
    train()