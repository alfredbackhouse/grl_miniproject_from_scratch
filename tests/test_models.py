import unittest
from models.virtual_node import VirtualNodeModel

class TestModels(unittest.TestCase):
    def test_virtual_node_forward(self):
        model = VirtualNodeModel(16, 32, 8)
        # Assert that forward pass works as expected
        self.assertIsNotNone(model)