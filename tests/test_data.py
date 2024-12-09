import unittest
from data.utils import load_planetoid_dataset

class TestDataset(unittest.TestCase):
    def test_load_planetoid(self):
        dataset = load_planetoid_dataset(name="Cora")
        data = dataset[0]
        self.assertIsNotNone(data)
        self.assertTrue(data.x.size(1) > 0)  # Check feature dimensions
        self.assertTrue(data.edge_index.size(1) > 0)  # Check edges
        self.assertTrue(data.y.size(0) > 0)  # Check labels

if __name__ == "__main__":
    unittest.main()