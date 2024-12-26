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
        self.analyzer.current_namespace = ["shopping_cart", "ShoppingCart", "add_item"]
        self.analyzer.current_class = "ShoppingCart"

    def test_resolve_self_method_call(self):
        """Test resolving a method call on self (self.calculate_total())"""
        # Find the self.calculate_total() call in add_item method
        add_item_node = self._get_function_node("shopping_cart.ShoppingCart.add_item")
        call_nodes = self._find_nodes(add_item_node, "call")
        attribute_node = call_nodes[1].children[0]  # The self.calculate_total part

        result = self.analyzer._resolve_attribute_call(attribute_node)
        self.assertEqual(result, "shopping_cart.ShoppingCart.calculate_total")

    def test_resolve_instance_method_call(self):
        """Test resolving a method call on an instance (cart.add_item())"""
        # Find the cart.add_item() call in process_shopping_cart
        process_func = self._get_function_node("shopping_cart.process_shopping_cart")
        call_nodes = self._find_nodes(process_func, "call")
        attribute_node = call_nodes[1].children[0]  # The cart.add_item part

        result = self.analyzer._resolve_attribute_call(attribute_node)
        self.assertEqual(result, "cart.add_item")

    def test_resolve_builtin_method_call(self):
        """Test resolving a method call on a built-in type (self.items.append())"""
        # Find the self.items.append() call in add_item method
        class_node = self._get_function_node("shopping_cart.ShoppingCart.add_item")
        append_call = self._find_nodes(class_node, "call")
        attribute_node = append_call[0].children[0]  # The self.items.append part

        result = self.analyzer._resolve_attribute_call(attribute_node)
        self.assertEqual(result, "self.items.append")


if __name__ == "__main__":
    unittest.main()
