import unittest
from data.utils import load_planetoid_dataset, hierarchical_clustering, hierarchical_reverse_clustering
from torch_geometric.data import Data
import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class TestGraphOperations(unittest.TestCase):
    # def test_load_planetoid(self):
    #     dataset = load_planetoid_dataset(name="Cora")
    #     data = dataset[0]
    #     self.assertIsNotNone(data)
    #     self.assertTrue(data.x.size(1) > 0)  # Check feature dimensions
    #     self.assertTrue(data.edge_index.size(1) > 0)  # Check edges
    #     self.assertTrue(data.y.size(0) > 0)  # Check labels

    def test_hierarchical_clustering_with_virtual_nodes(self):
        # Create a small test graph with 9 nodes (3 connected triangles)
        edge_index = torch.tensor([
            [0, 1], [1, 2], [2, 0],  # First triangle
            [3, 4], [4, 5], [5, 3],  # Second triangle
            [6, 7], [7, 8], [8, 6],  # Third triangle
            [2, 3], [5, 6]           # Connections between triangles
        ]).t()  # Transpose to match PyTorch Geometric format

        x = torch.tensor([
            [0, 0], [1, 0], [0.5, 0.866],  # First triangle
            [2, 0], [3, 0], [2.5, 0.866],  # Second triangle
            [4, 0], [5, 0], [4.5, 0.866]   # Third triangle
        ], dtype=torch.float)

        test_graph = Data(x=x, edge_index=edge_index)
        test_graph.train_mask = torch.ones(x.size(0), dtype=torch.bool)
        test_graph.val_mask = torch.zeros(x.size(0), dtype=torch.bool)
        test_graph.test_mask = torch.zeros(x.size(0), dtype=torch.bool)
        test_graph.y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=torch.long)


        num_clusters_per_level = [3, 1]
        num_levels = 2
        # Cluster assignments for each layer
        cluster_assignments = [
            [0, 0, 0, 1, 1, 1, 2, 2, 2],  # Layer 0 clusters (example: triangles)
            [0, 1, 2],                    # Layer 1 clusters
            [0, 1, 2]                     # Layer 2 clusters
        ]
        # Apply the hierarchical clustering function
        hierarchical_graph = hierarchical_clustering(
            data=test_graph,
            num_clusters_per_level=num_clusters_per_level,
            num_levels=num_levels
        )

        hierarchical_reverse_graph = hierarchical_reverse_clustering(
            data=test_graph,
            num_clusters_per_level=num_clusters_per_level,
            num_levels=num_levels
        )

        # Verify the structure of the hierarchical graph
        self.assertIsNotNone(hierarchical_graph)
        self.assertTrue(hierarchical_graph.x.size(0) > test_graph.x.size(0))  # More nodes due to virtual nodes
        self.assertTrue(hierarchical_graph.edge_index.size(1) > test_graph.edge_index.size(1))  # More edges

        # Verify the structure of the hierarchical unet graph 
        self.assertIsNotNone(hierarchical_reverse_graph)
        self.assertTrue(hierarchical_reverse_graph.x.size(0) > test_graph.x.size(0))
        self.assertTrue(hierarchical_reverse_graph.edge_index.size(1) > test_graph.edge_index.size(1))

        # Visualise the hierarchical graph
        def visualize_graph(data, title):
            g = nx.Graph()
            g.add_edges_from(data.edge_index.t().tolist())
            pos = {i: data.x[i].tolist() if i < test_graph.x.size(0) else [5 + i, 5 - i]
                   for i in range(data.x.size(0))}
            nx.draw(g, pos, with_labels=True, node_size=500, node_color="lightgreen")
            plt.title(title)
            plt.savefig(f"{title}.png")
            plt.show()

        def visualize_hierarchical_graph(data, title, num_original_nodes, layer_offsets):
            """
            Visualise the hierarchical graph with a layered layout, preserving triangular layouts.

            Args:
                data (torch_geometric.data.Data): The hierarchical graph.
                num_original_nodes (int): Number of nodes in the original graph.
                layer_offsets (list): List of offsets for virtual nodes in each layer.
                cluster_assignments (list): Cluster assignments for nodes in each layer.
            """
            g = nx.Graph()
            g.add_edges_from(data.edge_index.t().tolist())

            # Define positioning
            pos = {i: data.x[i].tolist() if i < test_graph.x.size(0) else [5 + i, 5 - i]
                   for i in range(data.x.size(0))}
            layer_gap = 5  # Vertical gap between layers

            # Virtual nodes in each layer
            for layer_idx, offset in enumerate(layer_offsets):
                cluster_start = offset[0]
                cluster_end = offset[1]
                num_clusters = cluster_end - cluster_start
                for cluster_id in range(num_clusters):
                    virtual_node_idx = cluster_start + cluster_id
                    pos[virtual_node_idx] = (pos[virtual_node_idx][0], pos[virtual_node_idx][1] -layer_gap * (layer_idx + 1))

            # Draw the graph
            plt.figure(figsize=(12, 8))
            print(len(pos))
            print(data.x.size(0))
            print(g)
            nx.draw(
                g, pos, with_labels=True, node_size=500,
                node_color=["lightblue" if i < num_original_nodes else "lightgreen" for i in range(data.x.size(0))],
                edge_color="gray"
            )
            plt.title(title)
            plt.savefig(f"{title}.png")
            plt.show()

        print("Visualising original graph...")
        visualize_graph(test_graph, "Original Graph")

        print("Visualising hierarchical graph...")
        print(hierarchical_graph)
        visualize_hierarchical_graph(hierarchical_graph, "Hierarchical Graph", num_original_nodes=9, layer_offsets=[(0, 9), (9, 12), (12, 13)])    

        print("Visualising hierarchical graph...")
        print(hierarchical_reverse_graph)
        visualize_hierarchical_graph(hierarchical_reverse_graph, "Hierarchical Reverse Graph", num_original_nodes=9, layer_offsets=[(0, 9), (9, 12), (12, 13), (13, 16)])    


if __name__ == "__main__":
    print('running test data')
    unittest.main()