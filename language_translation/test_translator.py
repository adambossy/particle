import unittest
from pathlib import Path

from tree_sitter import Node

from .translator import CallGraphAnalyzer


class TestCallGraphAnalyzerBase(unittest.TestCase):
    """Base class for CallGraphAnalyzer tests with common setup"""

    @classmethod
    def setUpClass(cls):
        # Sample code from shopping_cart.py
        cls.sample_code = """def format_price(amount):
    return "${amount:.2f}"


class ShoppingCart:
    def __init__(self):
        self.items = []

    def add_item(self, price):
        self.items.append(price)
        self.calculate_total()

    def calculate_total(self):
        total = sum(self.items)
        return format_price(total)


def process_shopping_cart():
    cart = ShoppingCart()
    cart.add_item(10.99)
    return cart
"""
        # Initialize the analyzer
        cls.analyzer = CallGraphAnalyzer(language="python")
        cls.analyzer.code = cls.sample_code.encode("utf-8")

        # Parse the code to get the AST
        cls.tree = cls.analyzer.parser.parse(cls.analyzer.code)

    def _find_node(self, root: Node, type_name: str) -> Node:
        """Helper method to find first node of given type in the AST"""
        if root.type == type_name:
            return root

        for child in root.children:
            result = self._find_node(child, type_name)
            if result:
                return result

        return None


class TestGetNodeText(TestCallGraphAnalyzerBase):
    """Test cases for the _get_node_text method"""

    def test_get_node_text_function_name(self):
        """Test getting function name text from AST"""
        # Find the format_price function node
        function_node = self._find_node(self.tree.root_node, "function_definition")
        identifier_node = self._find_node(function_node, "identifier")

        result = self.analyzer._get_node_text(identifier_node)
        self.assertEqual(result, "format_price")

    def test_get_node_text_class_name(self):
        """Test getting class name text from AST"""
        class_node = self._find_node(self.tree.root_node, "class_definition")
        identifier_node = self._find_node(class_node, "identifier")

        result = self.analyzer._get_node_text(identifier_node)
        self.assertEqual(result, "ShoppingCart")

    def test_get_node_text_method_name(self):
        """Test getting method name text from AST"""
        class_node = self._find_node(self.tree.root_node, "class_definition")
        method_node = self._find_node(class_node, "function_definition")
        identifier_node = self._find_node(method_node, "identifier")

        result = self.analyzer._get_node_text(identifier_node)
        self.assertEqual(result, "__init__")

    def test_get_node_text_empty_node(self):
        """Test getting text from None node"""
        result = self.analyzer._get_node_text(None)
        self.assertEqual(result, "")

    def test_get_node_text_string_literal(self):
        """Test getting text from string literal node"""
        # Find the string literal in format_price function
        function_node = self._find_node(self.tree.root_node, "function_definition")
        string_node = self._find_node(function_node, "string")

        result = self.analyzer._get_node_text(string_node)
        self.assertEqual(result, '"${amount:.2f}"')


if __name__ == "__main__":
    unittest.main()
