import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import GCNNorm
import torch_geometric

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