import builtins
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

import click
import tree_sitter_python
from graphviz import Digraph
from tree_sitter import Language, Node, Parser

# Define paths for both Swift and Python language libraries
LANGUAGE_PATHS = {"swift": "tree-sitter-swift/", "python": "tree-sitter-python/"}

# Load languages
LANGUAGES: Dict[str, Language] = {
    "python": Language(tree_sitter_python.language()),
}

# Map language names to file extensions
LANGUAGE_EXTENSIONS = {"python": ".py", "swift": ".swift"}


@dataclass
class FunctionInfo:
    """Store information about a function/method"""

    name: str
    namespace: str  # Full namespace path
    calls: Set[str]  # Set of fully qualified function names this function calls
    called_by: Set[str]  # Set of fully qualified function names that call this function
    start_point: tuple  # Line, column where function starts
    end_point: tuple  # Line, column where function ends
    node: Node  # AST node that was used to create this function


class CallGraphAnalyzer:
    def __init__(
        self, language: str, project_path: str = None, files: list[str] = None
    ):
        self.language = language.lower()
        self.project_path = Path(project_path) if project_path else None
        self.files = [Path(f) for f in files] if files else None

        if self.language not in LANGUAGES:
            raise ValueError(
                f"Unsupported language: {language}. Supported languages: {list(LANGUAGES.keys())}"
            )

        # Initialize parser
        self.parser = Parser(LANGUAGES[self.language])

        # Add new attributes for call graph
        self.functions: Dict[str, FunctionInfo] = {}  # Maps full_name -> FunctionInfo
        self.current_namespace: List[str] = (
            []
        )  # Track current namespace during traversal
        self.current_class = None  # Track current class during traversa
        self.tree = None  # Store the current AST
        self.code = None  # Store the current file's code

    def analyze(self):
        """Analyze either the project directory or specific files."""
        if self.project_path:
            self.analyze_project()
        elif self.files:
            self.analyze_files()
        else:
            raise ValueError("Either project_path or files must be provided")

    def analyze_files(self):
        """Analyze specific files and build call graph."""
        self.functions.clear()
        for file_path in self.files:
            # Use the helper function to check if the file is a test file
            if self._is_test_file(file_path):
                print(f"Skipping test file: {file_path}")
                continue

            print(f"\nAnalyzing file: {file_path}")
            self.current_namespace = [Path(file_path).stem]  # Start with module name
            ast = self.parse_file(str(file_path))
            self.collect_functions(ast)

    def _is_test_file(self, file_path: Path) -> bool:
        """Determine if a file is a test file based on its name."""
        return file_path.name.startswith("test_") or file_path.name.endswith("_test.py")

    def collect_functions(self, node: Node):
        """Build the call graph by traversing the AST."""
        if node.type == "function_definition":
            # Skip test functions
            if self._is_test_function(node):
                print(
                    f"Skipping test function: {self._get_symbol_name(self._find_identifier(node))}"
                )
                return
            self._process_function_definition(node)
        elif node.type == "class_definition":
            # Skip test classes
            if self._is_test_class(node):
                print(
                    f"Skipping test class: {self._get_symbol_name(self._find_identifier(node))}"
                )
                return
            self._process_class_definition(node)
        elif node.type == "call":
            self._process_function_call(node)

        for child in node.children:
            self.collect_functions(child)

        # Pop namespace when leaving class or function
        if node.type in ("function_definition", "class_definition"):
            self.current_namespace.pop()
            if node.type == "class_definition":
                self.current_class = None

    def _is_test_function(self, node: Node) -> bool:
        """Determine if a function is a test function based on its name."""
        identifier_node = self._find_identifier(node)
        if identifier_node:
            func_name = self._get_symbol_name(identifier_node)
            return func_name.startswith("test_")
        return False

    def _is_test_class(self, node: Node) -> bool:
        """Determine if a class is a test class based on its name."""
        identifier_node = self._find_identifier(node)
        if identifier_node:
            class_name = self._get_symbol_name(identifier_node)
            return class_name.startswith("Test")
        return False

    def _process_function_definition(self, node: Node):
        """Process a function definition node."""
        func_name = self._get_symbol_name(self._find_identifier(node))
        self.current_namespace.append(func_name)

        full_name = ".".join(self.current_namespace)
        function_info = self._get_or_create_function_info(func_name, full_name, node)
        function_info.start_point = node.start_point
        function_info.end_point = node.end_point

    def _get_or_create_function_info(
        self, func_name: str, full_name: str, node: Node
    ) -> FunctionInfo:
        if full_name in self.functions:
            return self.functions[full_name]

        self.functions[full_name] = FunctionInfo(
            name=func_name,
            namespace=full_name,
            calls=set(),
            called_by=set(),
            start_point=node.start_point,
            end_point=node.end_point,
            node=node,
        )

        return self.functions[full_name]

    def _process_class_definition(self, node: Node):
        """Process a class definition node."""
        class_name = self._get_symbol_name(self._find_identifier(node))
        self.current_namespace.append(class_name)
        self.current_class = class_name

    def _process_function_call(self, node: Node):
        """Process a function call node and update the call graph."""
        if not self.current_namespace:
            return  # Skip if we're not in any namespace

        caller_namespace = ".".join(self.current_namespace)
        if caller_namespace not in self.functions:
            return  # Skip if we're not in a function

        callee, full_callee = self._resolve_call(node)
        if callee:
            self.functions[caller_namespace].calls.add(full_callee)
            callee_info = self._get_or_create_function_info(callee, full_callee, node)
            callee_info.called_by.add(caller_namespace)

    def _resolve_call(self, node: Node) -> tuple[str, str]:
        """Resolve the full namespace of a function call."""
        if node.type != "call":
            return None, None

        # Get the function being called (first child of call node)
        func = node.children[0]

        if func.type == "identifier":
            return self._resolve_simple_call(func)
        elif func.type == "attribute":
            return self._resolve_attribute_call(func)
        elif func.type == "call":
            return self._resolve_nested_call(func)

        return None, None

    def _resolve_simple_call(self, func_node: Node) -> tuple[str, str]:
        """Handle simple function calls like my_function()"""
        func_name = self._get_symbol_name(func_node)

        # Check if it's a built-in function
        if func_name in dir(builtins):
            return func_name, f"builtins.{func_name}"

        # Imperfect way of grabbing the class name for class instantiations
        candidate_class_names = [
            node_name.split(".")[-2]
            for node_name in self.functions.keys()
            if len(node_name.split(".")) > 1
        ]
        if func_name in candidate_class_names:
            class_name = func_name
            return "__init__", f"{self.current_namespace[0]}.{class_name}.__init__"

        # Handle module.function() calls
        return func_name, f"{self.current_namespace[0]}.{func_name}"

    def _resolve_attribute_call(self, func_node: Node) -> tuple[str, str]:
        """Handle attribute-based calls like obj.method() or module.function()"""
        obj = func_node.children[0]
        method = func_node.children[2]

        obj_name = self._get_symbol_name(obj)
        method_name = self._get_symbol_name(method)

        # Handle self.method() calls
        if obj_name == "self" and self.current_class:
            return method_name, f"{'.'.join(self.current_namespace[:-1])}.{method_name}"

        # Handle class.static_method() calls
        if self.current_class and obj_name == self.current_class:
            return method_name, f"{'.'.join(self.current_namespace[:-1])}.{method_name}"

        # Handle module.function() calls
        if obj_name in self.current_namespace:
            return method_name, f"{obj_name}.{method_name}"

        # Handle instance.method() calls by trying to determine the object's type
        obj_type = self._infer_object_type(obj_name)
        if obj_type:
            return method_name, f"{obj_type}.{method_name}"

        # Fallback: return just obj_name.method_name
        return method_name, f"{obj_name}.{method_name}"

    def _infer_object_type(self, obj_name: str) -> str:
        """Attempt to infer the type of an object from the current context.

        This looks for:
        1. Variable assignments like: obj = ClassName()
        2. Function parameters with type hints
        3. Variable annotations

        Returns the fully qualified class name if found, None otherwise.
        """
        # Look for assignment in the current function's scope
        if self.current_namespace:
            current_func = self.functions.get(".".join(self.current_namespace))
            if current_func and current_func.node:
                # Search for assignment statements
                assignments = self._find_nodes(current_func.node, "assignment")
                for assign in assignments:
                    # Check if left side matches our object name
                    left_side = assign.children[0]
                    if (
                        left_side.type == "identifier"
                        and self._get_symbol_name(left_side) == obj_name
                    ):
                        # Look at right side for class instantiation
                        right_side = assign.children[2]
                        if right_side.type == "call":
                            class_name = self._get_symbol_name(right_side.children[0])
                            # Check if it's in current namespace
                            current_module = self.current_namespace[0]
                            return f"{current_module}.{class_name}"

        return None

    def _find_nodes(self, root: Node, type_name: str) -> list[Node]:
        """Find all nodes of a given type in the AST subtree."""
        nodes = []
        if root.type == type_name:
            nodes.append(root)
        for child in root.children:
            nodes.extend(self._find_nodes(child, type_name))
        return nodes

    def _resolve_nested_call(self, func_node: Node) -> tuple[str, str]:
        """Handle nested calls like get_object().method()"""
        # For nested calls, we focus on the final method being called
        # This is a simplified implementation
        if func_node.children and func_node.children[-1].type == "attribute":
            return self._resolve_attribute_call(func_node.children[-1])
        return None, None

    def _is_builtin_type(self, type_name: str) -> bool:
        """Check if a type is a Python built-in type."""
        return type_name.lower() in {
            "str",
            "int",
            "float",
            "list",
            "dict",
            "set",
            "tuple",
            "bool",
            "bytes",
            "object",
        }

    def _find_identifier(self, node: Node) -> Node:
        """Find the identifier node in a definition."""
        for child in node.children:
            if child.type == "identifier":
                return child

        return None

        # This handles cases like self.calculate_total() but not self.items.append(
        # because of the extra attribute present
        # return self._find_identifier_recursive(node)[1]

    # def _find_identifier_recursive(self, node: Node) -> List[Node]:
    #     """Find the identifier node in a definition."""
    #     if node.type == "identifier":
    #         return [node]

    #     identifiers = []
    #     for child in node.children:
    #         identifiers.extend(self._find_identifier_recursive(child))

    #     return identifiers

    def _get_symbol_name(self, node: Node) -> str:
        """Get the text content of a node."""
        if not node:
            return ""
        return self.code[node.start_byte : node.end_byte].decode("utf-8")

    def parse_file(self, file_path: str) -> Node:
        """Parse a single file and return its AST."""
        with open(file_path, "rb") as f:
            self.code = f.read()  # Store code for _get_symbol_name
        self.tree = self.parser.parse(self.code)  # Store the tree
        return self.tree.root_node

    def print_ast(self, node: Node, level: int = 0):
        """Recursively print the AST in a readable format."""
        node_type = str(node.type)
        indent = self._get_indent(level)
        print(f"{indent}{node_type}")
        self._print_children(node, level)

    def _get_indent(self, level: int) -> str:
        """Generate the appropriate indentation for the current level."""
        return "  " * level

    def _print_children(self, node: Node, level: int):
        """Print all children nodes recursively."""
        for child in node.children:
            self.print_ast(child, level + 1)

    def print_call_graph(self):
        """Print the generated call graph."""
        print("\nCall Graph Analysis:")
        for full_name, info in self.functions.items():
            print(f"\nFunction: {full_name}")
            if info.calls:
                print("  Calls:")
                for call in sorted(info.calls):
                    print(f"    → {call}")
            if info.called_by:
                print("  Called by:")
                for caller in sorted(info.called_by):
                    print(f"    ← {caller}")

    def analyze_project(self):
        """Analyze all files in the project directory."""
        # Get the file extension for the current language
        extension = LANGUAGE_EXTENSIONS.get(self.language, f".{self.language}")

        for file_path in self.project_path.rglob(f"*{extension}"):
            print(f"\nAnalyzing file: {file_path}")
            ast = self.parse_file(str(file_path))
            self.collect_functions(ast)

    def get_leaf_nodes(self) -> List[FunctionInfo]:
        """Return all FunctionInfo objects that don't call any other functions."""
        return [info for info in self.functions.values() if not info.calls]

    def get_nodes_at_level(self, level: int) -> List[FunctionInfo]:
        """Return all FunctionInfo objects at a specific level in the call tree.
        Level 0 represents leaf nodes (functions that don't call others).
        Level 1 represents functions that only call leaf nodes.
        Level 2 represents functions that call level 1 nodes, and so on.
        """
        if level < 0:
            return []

        # For level 0, return leaf nodes
        if level == 0:
            return self.get_leaf_nodes()

        all_prev_level_nodes = set()
        for level in range(level):
            all_prev_level_nodes.update(
                {node.namespace for node in self.get_nodes_at_level(level)}
            )

        # For other levels, find nodes that only call nodes from previous levels
        result = []
        for func in self.functions.values():
            # Skip if already found in previous levels
            if any(func.namespace in nodes for nodes in [all_prev_level_nodes]):
                continue

            # Check if all calls are to previous level nodes
            if func.calls and all(call in all_prev_level_nodes for call in func.calls):
                result.append(func)

        return result

    def visualize_graph(self, output_file: str = None, view: bool = True):
        """Create a visual representation of the call graph using graphviz.

        Args:
            output_file: Name of the output file (without extension)
            view: Whether to automatically open the generated graph
        """
        # Determine the output file name based on project_path or files
        if not output_file:
            if self.project_path:
                output_file = self.project_path.name
            elif self.files:
                file_names = [f.stem[:20] for f in self.files]
                output_file = "_".join(file_names)

        # Create a new directed graph
        dot = Digraph(comment="Call Graph")
        dot.attr(rankdir="TB")  # Top to bottom layout

        # Track processed nodes to avoid duplicates
        processed: Set[str] = set()

        # Helper function to add nodes and edges recursively
        def add_node_and_edges(namespace: str, processed: Set[str]):
            stack = [namespace]

            print(f"Recursing with namespace: {namespace}")

            while stack:
                current_namespace = stack.pop()

                print(f"Iterating with namespace: {current_namespace}")

                func_info = self.functions.get(current_namespace)
                if not func_info:
                    continue

                # Add the current node
                node_label = (
                    f"{func_info.name}\n({'.'.join(current_namespace.split('.')[:-1])})"
                )
                dot.node(current_namespace, node_label)
                processed.add(current_namespace)

                # Add edges for each function call
                for called_func in func_info.calls:
                    # Add the called function node if it hasn't been processed
                    if called_func not in processed:
                        called_info = self.functions.get(called_func)
                        if called_info:
                            called_label = f"{called_info.name}\n({'.'.join(called_func.split('.')[:-1])})"
                            dot.node(called_func, called_label)

                    # Add the edge
                    dot.edge(current_namespace, called_func)

                    # Add the called function to the stack for further processing
                    stack.append(called_func)

                processed.add(current_namespace)

        # Start with root nodes (functions that aren't called by others)
        root_nodes = {
            name for name, info in self.functions.items() if not info.called_by
        }

        # Process each root node
        for root in root_nodes:
            add_node_and_edges(root, processed)

        # Set graph attributes for better visualization
        dot.attr("node", shape="box", style="rounded")
        dot.attr("edge", arrowsize="0.5")

        # Save the graph
        dot.render(output_file, view=view, cleanup=True)
        print(f"Graph saved as {output_file}.pdf")


@click.command()
@click.option(
    "--project-path",
    type=click.Path(exists=True),
    help="Path to the project directory to analyze",
)
@click.option(
    "--files",
    type=click.Path(exists=True),
    multiple=True,
    help="One or more files to analyze",
)
@click.option(
    "--language",
    type=click.Choice(["python", "swift"], case_sensitive=False),
    default="python",
    help="Programming language to analyze",
)
def main(project_path, files, language):
    """Analyze Python or Swift code and build a call graph."""
    if not project_path and not files:
        raise click.UsageError("Either --project-path or --files must be provided")
    if project_path and files:
        raise click.UsageError("Cannot use both --project-path and --files together")

    analyzer = CallGraphAnalyzer(
        language=language,
        project_path=project_path if project_path else None,
        files=files if files else None,
    )
    analyzer.analyze()

    analyzer.print_ast(analyzer.tree.root_node, 0)
    analyzer.print_call_graph()

    print("Generating graph...")
    analyzer.visualize_graph()

    print("Done.")


if __name__ == "__main__":
    main()
