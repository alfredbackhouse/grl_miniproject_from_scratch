import torch
import yaml
from models.virtual_node import VirtualNodeModel
from models.baseline_gcn import BaselineGCNModel
from models.gcnii import GCNII
from models.gated_gcn import GatedGCN
import torch_geometric
from torch_geometric.loader import DataLoader
from data.utils import load_planetoid_dataset, load_processed_lrgb_dataset, load_lrgb_dataset, hierarchical_clustering, hierarchical_reverse_clustering
from sklearn.metrics import f1_score
from torch_geometric.utils import k_hop_subgraph, to_dense_adj
from sklearn.decomposition import PCA


# Convert edge_index to sparse adjacency matrix
def edge_index_to_sparse_matrix(edge_index, num_nodes):
    values = torch.ones(edge_index.size(1), device=edge_index.device)  # All edges have weight 1
    return torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))

def compute_rwse(edge_index, num_nodes, rwse_dim=16, walk_length=10):
    """
    Compute Random Walk Structural Embeddings (RWSE) for each node with reduced dimensionality.

    Args:
        edge_index: Edge index tensor of shape [2, num_edges].
        num_nodes: Number of nodes in the graph.
        rwse_dim: Dimensionality of the RWSE embeddings.
        walk_length: Number of steps for the random walk.

    Returns:
        rwse: Random Walk Structural Embeddings of shape [num_nodes, rwse_dim].
    """
    # Convert edge_index to adjacency matrix
    adj = to_dense_adj(edge_index)[0]  # Shape: [num_nodes, num_nodes]
    adj = adj / adj.sum(dim=1, keepdim=True)  # Row-normalise adjacency matrix

    # Compute powers of the adjacency matrix (random walks)
    rwse = [torch.eye(num_nodes, device=edge_index.device)]
    for _ in range(1, walk_length):
        rwse.append(rwse[-1] @ adj)
    rwse = torch.stack(rwse, dim=1)  # Shape: [num_nodes, walk_length, num_nodes]

    # Use steady-state probabilities (last dimension) as RWSE
    rwse = rwse[:, -1, :]  # Shape: [num_nodes, num_nodes]

    # Reduce dimensionality with PCA (applied to the node embeddings)
    rwse = rwse.cpu().numpy()  # Convert to numpy for PCA
    pca = PCA(n_components=rwse_dim)
    rwse_reduced = pca.fit_transform(rwse)  # Shape: [num_nodes, rwse_dim]
    rwse_reduced = torch.tensor(rwse_reduced, dtype=torch.float, device=edge_index.device)

    return rwse_reduced

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Read config file
with open("experiments/config.yaml", "r") as f:
    config = yaml.safe_load(f)
model_config = config["model"]
data_config = config["data"]
train_config = config["training"]

if data_config["dataset"] == "Cora":
    # Load the dataset
    train_dataset = load_planetoid_dataset(name=data_config["dataset"], root="./data/Planetoid")
    if model_config["clustering_type"] == "hsg":
        hiearchical_data = hierarchical_clustering(train_dataset[0], num_clusters_per_level=[16, 1], num_levels=2, cora=True)
    elif model_config["clustering_type"] == "unet":
        hiearchical_data = hierarchical_reverse_clustering(train_dataset[0], num_clusters_per_level=[16, 1], num_levels=2, unet=False, cora=True)
    elif model_config["clustering_type"] == "unet_with_skip":
        hiearchical_data = hierarchical_reverse_clustering(train_dataset[0], num_clusters_per_level=[16, 1], num_levels=2, unet=True, cora=True)
    elif model_config["clustering_type"] == "vn":
        hiearchical_data = hierarchical_clustering(train_dataset[0], num_clusters_per_level=[1], num_levels=1, cora=True)
    else: 
        print("No clustering type")
        hiearchical_data = train_dataset[0]
    train_dataset = [hiearchical_data]
else: 
    # Load the dataset (train split)
    train_dataset = load_processed_lrgb_dataset(name=data_config["dataset"], clustering_type=model_config["clustering_type"], split="train", root="./data/LRGB", overwrite=False)
    # Create a DataLoader for batching
    train_loader = DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True)
    # repeat for validation dataset
    val_dataset = load_processed_lrgb_dataset(name=data_config["dataset"], clustering_type=model_config["clustering_type"], split="val", root="./data/LRGB", overwrite=False)
    val_loader = DataLoader(val_dataset, batch_size=train_config["batch_size"], shuffle=True)

# Initialise model from config
model = BaselineGCNModel(
    input_dim=train_dataset[0].x.size(1) ,
    hid_dim=model_config["hid_dim"],
    n_classes=data_config["num_classes"],
    n_layers=model_config["n_layers"],
    dropout_ratio=model_config["dropout_ratio"]
).to(device)  # Move model to GPU
# model = GCNII(
#     nfeat=train_dataset[0].x.size(1),
#     nlayers=model_config["n_layers"],
#     nhidden=model_config["hid_dim"],
#     nclass=data_config["num_classes"],
#     dropout=model_config["dropout_ratio"],
#     lamda=0.5,
#     alpha=0.1,
#     variant=False
# ).to(device)
# model = GCNConv(train_dataset[0].x.size(1), model_config["hid_dim"]).to(device)
model.reset_parameters()

print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Collect all labels from the dataset
all_labels = torch.cat([data.y for data in train_dataset], dim=0)

# Count occurrences of each class
if data_config["dataset"] == "Cora":
    criterion = torch.nn.CrossEntropyLoss().to(device)
else: 
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
            # rwse = compute_rwse(batch.edge_index, batch.x.size(0), 16).to(device)
            batch.edge_index = k_hop_subgraph(torch.arange(batch.x.size(0), device=batch.edge_index.device), model_config["head_depth"], batch.edge_index, relabel_nodes=False)[1]
            batch = batch.to(device)  # Move batch to GPU
            optimizer.zero_grad()
            
            # Forward pass
            logits, embedding = model(batch.x, batch.edge_index)
            
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
            for batch in val_loader:  # Use validation DataLoader here if available
                # rwse = compute_rwse(batch.edge_index, batch.x.size(0), 16).to(device)
                batch.edge_index = k_hop_subgraph(torch.arange(batch.x.size(0), device=batch.edge_index.device), model_config["head_depth"], batch.edge_index, relabel_nodes=False)[1]
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

        # Calculate accuracy 
        val_acc = sum([1 for i in range(len(all_preds)) if all_preds[i] == all_labels[i]]) / len(all_preds)

        # Print metrics
        print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}, Val Loss = {val_loss:.4f}, Val F1-Score = {val_f1:.4f}, Val Accuracy = {val_acc:.4f}")

# Training loop
def train_cora(data):
    model.train()
    print(data[0])
    data = data[0].to(device)

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
            val_preds = val_logits[data.val_mask].argmax(dim=1).cpu().numpy()
            val_labels = data.y[data.val_mask].cpu().numpy()
            val_accuracy = (val_preds == val_labels).sum() / data.val_mask.sum()
            val_f1 = f1_score(val_labels, val_preds, average="macro")

        model.train()  # Switch back to training mode

        # Print metrics
        print(f"Epoch {epoch+1}: Train Loss = {train_loss.item():.4f}, Val Loss = {val_loss.item():.4f}, Val F1-Score = {val_f1:.4f}, Val Accuracy = {val_accuracy:.4f}")
              
if __name__ == "__main__":
    if data_config["dataset"] == "Cora":
        train_cora(train_dataset)
    else:
        train()


import torch
from torch_geometric.utils import to_dense_adj

