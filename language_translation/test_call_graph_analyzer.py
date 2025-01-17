import unittest
from pathlib import Path

from tree_sitter import Node

from language_translation.call_graph_analyzer import CallGraphAnalyzer


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
        # Create a temporary file with the sample code
        cls.temp_file = Path("shopping_cart.py")
        cls.temp_file.write_text(cls.sample_code)

        # Initialize and analyze
        cls.analyzer = CallGraphAnalyzer(language="python", files=[str(cls.temp_file)])
        cls.analyzer.analyze()

        # Parse the code to get the AST
        cls.tree = cls.analyzer.tree

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary file
        if cls.temp_file.exists():
            cls.temp_file.unlink()

    def _find_nodes(self, root: Node, type_name: str) -> list[Node]:
        """Helper method to find all nodes of given type in the AST"""
        nodes = []
        if root.type == type_name:
            nodes.append(root)

        for child in root.children:
            nodes.extend(self._find_nodes(child, type_name))

        return nodes

    def _find_node(self, root: Node, type_name: str) -> Node:
        """Helper method to find first node of given type in the AST"""
        if root.type == type_name:
            return root

        for child in root.children:
            result = self._find_node(child, type_name)
            if result:
                return result

        return None

    def _get_function_node(self, full_name: str) -> Node:
        """Get the function node for a given full name"""
        return self.analyzer.functions.get(full_name).node


class TestGetSymbolName(TestCallGraphAnalyzerBase):
    """Test cases for the _get_symbol_name method"""

    def test_get_symbol_name_function_name(self):
        """Test getting function name text from AST"""
        # Find the format_price function node
        function_node = self._find_node(self.tree.root_node, "function_definition")
        identifier_node = self._find_node(function_node, "identifier")

        result = self.analyzer._get_symbol_name(identifier_node)
        self.assertEqual(result, "format_price")

    def test_get_symbol_name_class_name(self):
        """Test getting class name text from AST"""
        class_node = self._find_node(self.tree.root_node, "class_definition")
        identifier_node = self._find_node(class_node, "identifier")

        result = self.analyzer._get_symbol_name(identifier_node)
        self.assertEqual(result, "ShoppingCart")

    def test_get_symbol_name_method_name(self):
        """Test getting method name text from AST"""
        class_node = self._find_node(self.tree.root_node, "class_definition")
        method_node = self._find_node(class_node, "function_definition")
        identifier_node = self._find_node(method_node, "identifier")

        result = self.analyzer._get_symbol_name(identifier_node)
        self.assertEqual(result, "__init__")

    def test_get_symbol_name_empty_node(self):
        """Test getting text from None node"""
        result = self.analyzer._get_symbol_name(None)
        self.assertEqual(result, "")

    def test_get_symbol_name_string_literal(self):
        """Test getting text from string literal node"""
        # Find the string literal in format_price function
        function_node = self._find_node(self.tree.root_node, "function_definition")
        string_node = self._find_node(function_node, "string")

        result = self.analyzer._get_symbol_name(string_node)
        self.assertEqual(result, '"${amount:.2f}"')


class TestFindIdentifier(TestCallGraphAnalyzerBase):
    """Test cases for the _find_identifier method"""

    def test_find_identifier_in_function(self):
        """Test finding identifier in function definition"""
        function_node = self._get_function_node("shopping_cart.format_price")
        identifier = self.analyzer._find_identifier(function_node)

        self.assertIsNotNone(identifier)
        self.assertEqual(identifier.type, "identifier")
        self.assertEqual(self.analyzer._get_symbol_name(identifier), "format_price")

    def test_find_identifier_in_class(self):
        """Test finding identifier in class definition"""
        class_node = self._find_node(self.tree.root_node, "class_definition")
        identifier = self.analyzer._find_identifier(class_node)

        self.assertIsNotNone(identifier)
        self.assertEqual(identifier.type, "identifier")
        self.assertEqual(self.analyzer._get_symbol_name(identifier), "ShoppingCart")

    def test_find_identifier_in_method(self):
        """Test finding identifier in method definition"""
        method_node = self._get_function_node("shopping_cart.ShoppingCart.__init__")
        identifier = self.analyzer._find_identifier(method_node)

        self.assertIsNotNone(identifier)
        self.assertEqual(identifier.type, "identifier")
        self.assertEqual(self.analyzer._get_symbol_name(identifier), "__init__")

    def test_find_identifier_in_invalid_node(self):
        """Test finding identifier in node that shouldn't have one"""
        # Find a string literal node which shouldn't have an identifier
        string_node = self._find_node(self.tree.root_node, "string")
        identifier = self.analyzer._find_identifier(string_node)

        self.assertIsNone(identifier)


class TestResolveAttributeCall(TestCallGraphAnalyzerBase):
    """Test cases for the _resolve_attribute_call method"""

    def setUp(self):
        """Set up the namespace context for each test"""
        self.analyzer.current_namespace = ["ShoppingCart", "add_item"]
        self.analyzer.current_class = "ShoppingCart"

    def test_resolve_self_method_call(self):
        """Test resolving a method call on self (self.calculate_total())"""
        # Find the self.calculate_total() call in add_item method
        add_item_node = self._get_function_node("shopping_cart.ShoppingCart.add_item")
        call_nodes = self._find_nodes(add_item_node, "call")
        attribute_node = call_nodes[1].children[0]  # The self.calculate_total part

        function_info = self.analyzer._resolve_attribute_call(attribute_node)
        self.assertEqual(function_info.module_name(), "shopping_cart")
        self.assertEqual(function_info.namespace, "ShoppingCart")
        self.assertEqual(function_info.name, "calculate_total")
        self.assertEqual(
            function_info.key(), "shopping_cart.ShoppingCart.calculate_total"
        )

    def test_resolve_instance_method_call(self):
        """Test resolving a method call on an instance (cart.add_item())"""
        # Find the cart.add_item() call in process_shopping_cart
        process_func = self._get_function_node("shopping_cart.process_shopping_cart")
        call_nodes = self._find_nodes(process_func, "call")
        attribute_node = call_nodes[1].children[0]  # The cart.add_item part

        self.analyzer.current_namespace = ["process_shopping_cart"]

        function_info = self.analyzer._resolve_attribute_call(attribute_node)
        self.assertEqual(function_info.module_name(), "shopping_cart")
        self.assertEqual(function_info.namespace, "ShoppingCart")
        self.assertEqual(function_info.name, "add_item")
        self.assertEqual(function_info.key(), "shopping_cart.ShoppingCart.add_item")

    def test_resolve_builtin_method_call(self):
        """Test resolving a method call on a built-in type (self.items.append())"""
        # Find the self.items.append() call in add_item method
        class_node = self._get_function_node("shopping_cart.ShoppingCart.add_item")
        append_call = self._find_nodes(class_node, "call")
        attribute_node = append_call[0].children[0]  # The self.items.append part

        function_info = self.analyzer._resolve_attribute_call(attribute_node)
        self.assertEqual(function_info.module_name(), "UNKNOWN")
        self.assertIsNone(function_info.namespace)
        self.assertEqual(function_info.name, "append")
        self.assertEqual(function_info.key(), "append")


class TestGetLeafNodes(TestCallGraphAnalyzerBase):
    """Test cases for the get_leaf_nodes method"""

    def test_get_leaf_nodes(self):
        """Test identifying functions that don't call other functions"""
        leaf_nodes = self.analyzer.get_leaf_nodes()

        # Convert to set of names for easier comparison
        leaf_names = {node.key() for node in leaf_nodes}

        # In our sample code, format_price and calculate_total are leaf nodes
        # as they don't call any other functions
        expected_leaves = {
            "append",
            "builtins.sum",
            "shopping_cart.ShoppingCart.__init__",
            "shopping_cart.format_price",
        }

        self.assertEqual(leaf_names, expected_leaves)

    def test_get_leaf_nodes_empty_code(self):
        """Test get_leaf_nodes with empty code"""
        # Create a new analyzer with empty code
        empty_code = """
class EmptyClass:
    pass
"""
        temp_file = Path("empty.py")
        temp_file.write_text(empty_code)

        try:
            analyzer = CallGraphAnalyzer(language="python", files=[str(temp_file)])
            analyzer.analyze()

            leaf_nodes = analyzer.get_leaf_nodes()
            self.assertEqual(len(leaf_nodes), 0)
        finally:
            if temp_file.exists():
                temp_file.unlink()


class TestGetNodesAtLevel(TestCallGraphAnalyzerBase):
    """Test cases for the get_nodes_at_level method"""

    def test_get_nodes_level_0(self):
        """Test getting leaf nodes (level 0)"""
        nodes = self.analyzer.get_nodes_at_level(0)
        node_names = {node.key() for node in nodes}

        # Level 0 should match leaf nodes
        expected_nodes = {
            "append",
            "builtins.sum",
            "shopping_cart.ShoppingCart.__init__",
            "shopping_cart.format_price",
        }

        self.assertEqual(node_names, expected_nodes)

    def test_get_nodes_level_1(self):
        """Test getting nodes that only call leaf nodes (level 1)"""
        nodes = self.analyzer.get_nodes_at_level(1)
        node_names = {node.key() for node in nodes}

        # calculate_total calls format_price (level 0)
        expected_nodes = {
            "shopping_cart.ShoppingCart.calculate_total",
        }

        self.assertEqual(node_names, expected_nodes)

    def test_get_nodes_level_2(self):
        """Test getting nodes that call level 1 nodes (level 2)"""
        nodes = self.analyzer.get_nodes_at_level(2)
        node_names = {node.key() for node in nodes}

        # add_item calls calculate_total (level 1)
        expected_nodes = {
            "shopping_cart.ShoppingCart.add_item",
        }

        self.assertEqual(node_names, expected_nodes)

    def test_get_nodes_level_3(self):
        """Test getting nodes that call level 2 nodes (level 3)"""
        nodes = self.analyzer.get_nodes_at_level(3)
        node_names = {node.key() for node in nodes}

        # process_shopping_cart calls add_item (level 2)
        expected_nodes = {
            "shopping_cart.process_shopping_cart",
        }

        self.assertEqual(node_names, expected_nodes)

    def test_get_nodes_invalid_level(self):
        """Test getting nodes at an invalid level"""
        # Test negative level
        nodes = self.analyzer.get_nodes_at_level(-1)
        self.assertEqual(len(nodes), 0)

        # Test level beyond max depth
        nodes = self.analyzer.get_nodes_at_level(10)
        self.assertEqual(len(nodes), 0)


class TestProcessImports(TestCallGraphAnalyzerBase):
    """Test cases for processing import statements"""

    @classmethod
    def setUpClass(cls):
        # Create a temporary file with import statements
        cls.import_code = """import nltk
import itertools as it
from shopping_cart import ShoppingCart, format_price
from . import models
from .models import GenAIImage, StoryDraft, StoryDraftState
from .tasks.image_gen import gen_images as gen_images_task
"""
        cls.temp_file = Path("temp_imports.py")
        cls.temp_file.write_text(cls.import_code)

        # Initialize and analyze
        cls.analyzer = CallGraphAnalyzer(language="python", files=[str(cls.temp_file)])
        cls.analyzer.analyze()

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary file
        if cls.temp_file.exists():
            cls.temp_file.unlink()

    def test_simple_import(self):
        """Test processing a simple import statement like 'import nltk'"""
        # Check if 'nltk' is correctly added to imports
        self.assertIn("nltk", self.analyzer.imports)
        self.assertEqual(self.analyzer.imports["nltk"], "nltk")

    def test_import_with_alias(self):
        """Test processing an import with alias like 'import itertools as it'"""
        self.assertIn("it", self.analyzer.imports)
        self.assertEqual(self.analyzer.imports["it"], "itertools")

    def test_from_import_multiple(self):
        """Test processing from-import with multiple names"""
        # Check if both imports from shopping_cart are present
        self.assertIn("ShoppingCart", self.analyzer.imports)
        self.assertEqual(
            self.analyzer.imports["ShoppingCart"], "shopping_cart.ShoppingCart"
        )

        self.assertIn("format_price", self.analyzer.imports)
        self.assertEqual(
            self.analyzer.imports["format_price"], "shopping_cart.format_price"
        )

    def test_relative_import(self):
        """Test processing relative import like 'from . import models'"""
        self.assertIn("models", self.analyzer.imports)
        self.assertEqual(self.analyzer.imports["models"], ".models")

    def test_relative_import_multiple(self):
        """Test processing relative import with multiple names"""
        self.assertIn("GenAIImage", self.analyzer.imports)
        self.assertEqual(self.analyzer.imports["GenAIImage"], "models.GenAIImage")

        self.assertIn("StoryDraft", self.analyzer.imports)
        self.assertEqual(self.analyzer.imports["StoryDraft"], "models.StoryDraft")

        self.assertIn("StoryDraftState", self.analyzer.imports)
        self.assertEqual(
            self.analyzer.imports["StoryDraftState"], "models.StoryDraftState"
        )

    def test_relative_import_with_alias(self):
        """Test processing relative import with alias"""
        self.assertIn("gen_images_task", self.analyzer.imports)
        self.assertEqual(
            self.analyzer.imports["gen_images_task"], "tasks.image_gen.gen_images"
        )


if __name__ == "__main__":
    unittest.main()
