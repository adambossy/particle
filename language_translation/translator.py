import builtins
from dataclasses import dataclass, field
from datetime import datetime
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
class TranslatorNode:
    """Base class for storing information about code entities like functions and classes"""

    name: str
    namespace: str  # Full namespace path
    node: Node  # AST node that was used to create this entity
    file: str  # Path to the file containing this entity
    class_deps: Set[str] = field(default_factory=set)
    var_deps: Set[str] = field(default_factory=set)
    source_code: str = ""  # Add source code field


@dataclass
class FunctionNode(TranslatorNode):
    """Store information about a function/method"""

    # The default values here are an ugly hack to get the parent dataclasses' default values to work
    start_point: tuple = (-1, -1)  # Line, column where function starts
    end_point: tuple = (-1, -1)  # Line, column where function ends
    calls: Set[str] = field(
        default_factory=set
    )  # Set of fully qualified function names this function calls
    called_by: Set[str] = field(
        default_factory=set
    )  # Set of fully qualified function names that call this function


@dataclass
class ClassNode(TranslatorNode):
    """Store information about a class"""

    # No additional fields needed for now, but can be extended in the future


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
        self.functions: Dict[str, FunctionNode] = {}  # Maps full_name -> FunctionInfo
        self.current_namespace: List[str] = (
            []
        )  # Track current namespace during traversal
        self.current_class = None  # Track current class during traversa
        self.tree = None  # Store the current AST
        self.code = None  # Store the current file's code
        self.imports = {}  # Maps local names to full module paths
        self.current_file = None  # Track the current file being analyzed
        self.classes: Dict[str, ClassNode] = {}  # Maps full_name -> ClassInfo

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
        # Add import handling
        if node.type == "import_statement":
            self._process_import_statement(node)
        elif node.type == "import_from_statement":
            self._process_import_from_statement(node)
        elif node.type == "function_definition":
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
        function_info = self._get_or_create_function_info(
            func_name, full_name, node, file=self.current_file
        )
        function_info.start_point = node.start_point
        function_info.end_point = node.end_point

    def _get_or_create_function_info(
        self,
        func_name: str,
        full_name: str,
        node: Node,
        file: str = "UNKNOWN",
    ) -> FunctionNode:
        function_info = self.functions.get(full_name)
        if function_info:
            if function_info.file == "UNKNOWN" and file != "UNKNOWN":
                function_info.file = file
            return function_info

        # Capture the source code using the node's byte range
        source_code = self.code[node.start_byte : node.end_byte].decode("utf-8")

        self.functions[full_name] = FunctionNode(
            name=func_name,
            namespace=full_name,
            calls=set(),
            called_by=set(),
            start_point=node.start_point,
            end_point=node.end_point,
            node=node,
            file=file,
            source_code=source_code,  # Store the source code
        )

        return self.functions[full_name]

    def _process_class_definition(self, node: Node):
        """Process a class definition node."""
        class_name = self._get_symbol_name(self._find_identifier(node))

        # Create full name using current namespace
        full_name = (
            f"{self.current_namespace[0]}.{class_name}"
            if self.current_namespace
            else class_name
        )

        # Store class info
        self.classes[full_name] = ClassNode(
            name=class_name,
            namespace=full_name,
            node=node,
            file=self.current_file,
        )

        # Continue with existing namespace tracking
        self.current_namespace.append(class_name)
        self.current_class = class_name

    def _process_function_call(self, node: Node):
        """Process a function call node and update the call graph."""
        if not self.current_namespace:
            return  # Skip if we're not in any namespace

        caller_namespace = ".".join(self.current_namespace)
        if caller_namespace not in self.functions:
            return  # Skip if we're not in a function

        callee_info = self._resolve_call(node)
        if isinstance(callee_info, FunctionNode):
            self.functions[caller_namespace].calls.add(callee_info.namespace)
            callee_info.called_by.add(caller_namespace)
        elif isinstance(callee_info, ClassNode):
            self.functions[caller_namespace].class_deps.add(callee_info.namespace)

            # Add the __init__ function as well for some redundancy, who knows if we'll need it later
            function_info = self._get_or_create_function_info(
                "__init__",
                callee_info.namespace + ".__init__",
                callee_info.node,
                callee_info.file,
            )
            self.functions[caller_namespace].calls.add(function_info.namespace)
            function_info.called_by.add(caller_namespace)

    def _resolve_call(self, node: Node) -> FunctionNode:
        """Resolve the full namespace of a function call."""
        if node.type != "call":
            return None

        # Get the function being called (first child of call node)
        func = node.children[0]

        if func.type == "identifier":
            return self._resolve_simple_call(func)
        elif func.type == "attribute":
            return self._resolve_attribute_call(func)
        elif func.type == "call":
            return self._resolve_nested_call(func)

        return None

    def _resolve_simple_call(self, func_node: Node) -> FunctionNode:
        """Handle simple function calls like my_function()"""
        func_name = self._get_symbol_name(func_node)

        # Check if it's a call to an imported module
        if func_name in self.imports:
            full_name = f"{self.imports[func_name]}"
            return self._get_or_create_function_info(func_name, full_name, func_node)

        # Check if it's a built-in function
        if func_name in dir(builtins):
            return self._get_or_create_function_info(
                func_name, f"builtins.{func_name}", func_node
            )

        # Imperfect way of grabbing the class name for class instantiations
        candidate_class_names = [
            node_name.split(".")[-2]
            for node_name in self.functions.keys()
            if len(node_name.split(".")) > 1
        ]
        if func_name in candidate_class_names:
            class_info = self._get_or_create_class_info(func_name, func_node)
            return class_info

        # Handle module.function() calls
        full_name = f"{self.current_namespace[0]}.{func_name}"
        return self._get_or_create_function_info(func_name, full_name, func_node)

    def _get_or_create_class_info(
        self, class_name: str, node: Node, file: str = "UNKNOWN"
    ) -> ClassNode:
        """Get or create a ClassInfo object."""
        full_name = f"{self.current_namespace[0]}.{class_name}"
        class_info = self.classes.get(full_name)
        if not class_info:
            # Capture the source code using the node's byte range
            source_code = self.code[node.start_byte : node.end_byte].decode("utf-8")

            class_info = ClassNode(
                name=class_name,
                namespace=full_name,
                node=node,
                file=file,
                source_code=source_code,  # Store the source code
            )
            self.classes[full_name] = class_info
        return class_info

    def _resolve_attribute_call(self, func_node: Node) -> FunctionNode:
        """Handle attribute-based calls like obj.method() or module.function()"""
        obj = func_node.children[0]
        method = func_node.children[2]

        obj_name = self._get_symbol_name(obj)
        obj_tokens = obj_name.split(".")
        method_name = self._get_symbol_name(method)

        # Handle calls on imported modules
        if obj_name in self.imports:
            base_module = self.imports[obj_name]
            full_name = f"{base_module}.{method_name}"
            return self._get_or_create_function_info(method_name, full_name, func_node)

        # Handle self.method() calls
        if obj_name == "self" and self.current_class:
            full_name = f"{'.'.join(self.current_namespace[:-1])}.{method_name}"
            return self._get_or_create_function_info(method_name, full_name, func_node)

        # Handle self.instance.method() calls
        if obj_name.startswith("self") and self.current_class:
            obj_type = self._infer_object_type(obj_name)
            if obj_type:
                full_name = f"{obj_type}.{method_name}"
                return self._get_or_create_function_info(
                    method_name, full_name, func_node
                )
            else:
                # Default to builtins
                full_name = f"builtins.{method_name}"
                return self._get_or_create_function_info(
                    method_name, full_name, func_node
                )

        # Handle class.static_method() calls
        if self.current_class and obj_name == self.current_class:
            full_name = f"{'.'.join(self.current_namespace[:-1])}.{method_name}"
            return self._get_or_create_function_info(method_name, full_name, func_node)

        # Handle module.function() calls
        if obj_name in self.current_namespace:
            full_name = f"{obj_name}.{method_name}"
            return self._get_or_create_function_info(method_name, full_name, func_node)

        # Handle instance.method() calls by trying to determine the object's type
        obj_type = self._infer_object_type(obj_name)
        if obj_type:
            full_name = f"{obj_type}.{method_name}"
            return self._get_or_create_function_info(method_name, full_name, func_node)

        # Fallback: return just obj_name.method_name
        full_name = f"{obj_name}.{method_name}"
        return self._get_or_create_function_info(method_name, full_name, func_node)

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

    def _resolve_nested_call(self, func_node: Node) -> FunctionNode:
        """Handle nested calls like get_object().method()"""
        # For nested calls, we focus on the final method being called
        # This is a simplified implementation
        if func_node.children and func_node.children[-1].type == "attribute":
            return self._resolve_attribute_call(func_node.children[-1])
        return None

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
        self.current_file = file_path  # Set current file
        with open(file_path, "rb") as f:
            self.code = f.read()  # Store code for _get_symbol_name
        self.tree = self.parser.parse(self.code)  # Store the tree
        return self.tree.root_node

    def print_ast(
        self,
        node: Node,
        level: int = 0,
        output_file: str = None,
        output_lines: list = None,
    ):
        """Recursively print the AST in a readable format and save to log file."""
        # Initialize output_lines list on first call
        if output_lines is None:
            output_lines = ["Abstract Syntax Tree:"]
            log_path = self._get_log_path(output_file, "ast")
            is_root_call = True
        else:
            is_root_call = False

        node_type = str(node.type)
        indent = self._get_indent(level)
        line = f"{indent}{node_type}"

        # Add to output lines and print to stdout
        output_lines.append(line)

        # Recursively process children
        for child in node.children:
            self.print_ast(child, level + 1, output_file, output_lines)

        # Write to log file if this is the root call
        if is_root_call:
            log_path = self._get_log_path(output_file, "ast")
            with open(log_path, "w") as f:
                f.write("\n".join(output_lines))
            print(f"\nAST log saved to: {log_path}")

    def _get_indent(self, level: int) -> str:
        """Generate the appropriate indentation for the current level."""
        return "  " * level

    def _get_log_path(
        self, output_file: str = None, log_type: str = "call_graph"
    ) -> Path:
        """Generate the log file path with timestamp.

        Args:
            output_file: Base name for the output directory
            log_type: Type of log file ("call_graph" or "ast")
        """
        output_file = self._output_file_slug(output_file)
        timestamp = datetime.now().strftime(
            "%Y-%m-%d_%I:%M_%p"
        )  # e.g. 2024-03-20_02:30_PM
        log_dir = Path("logs") / f"{output_file}_{timestamp}"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / f"{log_type}.log"

    def print_call_graph(self, output_file: str = None):
        """Print the generated call graph to both stdout and a log file."""
        log_path = self._get_log_path(output_file, "call_graph")

        # Create a list to store the output lines
        output_lines = ["Call Graph Analysis:"]
        for full_name, info in self.functions.items():
            # Include the file name in the output
            output_lines.append(f"\nFunction: {full_name}")
            output_lines.append(f"  File: {info.file}")
            if info.calls:
                output_lines.append("  Calls:")
                for call in sorted(info.calls):
                    output_lines.append(f"    → {call}")
            if info.called_by:
                output_lines.append("  Called by:")
                for caller in sorted(info.called_by):
                    output_lines.append(f"    ← {caller}")

        # Write to log file
        with open(log_path, "w") as f:
            f.write("\n".join(output_lines))

        print(f"\nCall graph log saved to: {log_path}")

    def analyze_project(self):
        """Analyze all files in the project directory."""
        # Get the file extension for the current language
        extension = LANGUAGE_EXTENSIONS.get(self.language, f".{self.language}")

        for file_path in self.project_path.rglob(f"*{extension}"):
            print(f"\nAnalyzing file: {file_path}")
            ast = self.parse_file(str(file_path))
            self.collect_functions(ast)

    def get_leaf_nodes(self) -> List[FunctionNode]:
        """Return all FunctionInfo objects that don't call any other functions."""
        return [info for info in self.functions.values() if not info.calls]

    def get_nodes_at_level(self, level: int) -> List[FunctionNode]:
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

    def _output_file_slug(self, output_file: str = None) -> str:
        # Determine the output file name based on project_path or files
        if not output_file:
            if self.project_path:
                output_file = self.project_path.name
            elif self.files:
                file_names = [f.stem[:20] for f in self.files]
                output_file = "_".join(file_names)

        return output_file

    def visualize_graph(self, output_file: str = None, view: bool = True):
        """Create a visual representation of the call graph using graphviz.

        Args:
            output_file: Name of the output file (without extension)
            view: Whether to automatically open the generated graph
        """
        output_file = self._output_file_slug(output_file)

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

    def _process_import_statement(self, node: Node):
        """Process a simple import statement like 'import foo' or 'import foo as bar'."""
        for child in node.children:
            if child.type == "dotted_name":
                module_path = self._get_symbol_name(child)
                alias = module_path  # Default alias is the module path itself

                # Check for 'as' alias
                next_sibling = child.next_sibling
                if next_sibling and next_sibling.type == "as":
                    alias_node = next_sibling.next_sibling
                    if alias_node:
                        alias = self._get_symbol_name(alias_node)

                self.imports[alias] = module_path

    def _process_import_from_statement(self, node: Node):
        """Process from-import statements like 'from foo import bar' or 'from foo import bar as baz'."""
        # Get the module path (after 'from')
        module_node = None
        for child in node.children:
            if child.type == "dotted_name":
                module_node = child
                break

        if not module_node:
            return

        module_path = self._get_symbol_name(module_node)

        # Process imported names
        for child in node.children:
            if child.type == "import_statement":
                for import_child in child.children:
                    if import_child.type == "dotted_name":
                        name = self._get_symbol_name(import_child)
                        alias = name  # Default alias is the name itself

                        # Check for 'as' alias
                        next_sibling = import_child.next_sibling
                        if next_sibling and next_sibling.type == "as":
                            alias_node = next_sibling.next_sibling
                            if alias_node:
                                alias = self._get_symbol_name(alias_node)

                        self.imports[alias] = f"{module_path}.{name}"


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
