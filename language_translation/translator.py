import ast
import builtins
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import click
from graphviz import Digraph
from tree_sitter import Language, Node, Parser

# # Define paths for both Swift and Python language libraries
# LANGUAGE_PATHS = {"swift": "tree-sitter-swift/", "python": "tree-sitter-python/"}

# # Load languages
# LANGUAGES: Dict[str, Language] = {
#     "python": Language(tree_sitter_python.language()),
# }

# # Map language names to file extensions
# LANGUAGE_EXTENSIONS = {"python": ".py", "swift": ".swift"}


def get_function_key(func_name: str, namespace: str | None, file_path: str) -> str:
    module_name = get_module_name(file_path)
    tokens = []
    if module_name != "UNKNOWN":
        tokens.append(module_name)
    if namespace:
        tokens.append(namespace)
    tokens.append(func_name)
    return ".".join(tokens)


def get_module_name(file_path: str) -> str:
    return Path(file_path).stem


@dataclass
class TranslatorNode:
    """Base class for storing information about code entities like functions and classes"""

    name: str
    node: Node  # AST node that was used to create this entity
    file: str  # Path to the file containing this entity
    class_deps: Set[str] = field(default_factory=set)
    var_deps: Set[str] = field(default_factory=set)
    source_code: str = ""  # Add source code field
    is_test: bool = False  # Add is_test field

    def module_name(self) -> str:
        """Return the module name for the node."""
        return get_module_name(self.file)

    def __str__(self) -> int:
        """Return the hash of the node, using its unique key."""
        return self.key()


@dataclass
class FunctionNode(TranslatorNode):
    """Store information about a function/method"""

    # The default values here are an ugly hack to get the parent dataclasses' default values to work
    namespace: str | None = None  # Full namespace path
    start_point: tuple = (-1, -1)  # Line, column where function starts
    end_point: tuple = (-1, -1)  # Line, column where function ends
    calls: Set["FunctionNode"] = field(
        default_factory=set
    )  # Set of fully qualified function names this function calls
    called_by: Set[str] = field(
        default_factory=set
    )  # Set of fully qualified function names that call this function

    def key(self) -> str:
        """Return a unique key for the node, combining namespace and name."""
        return get_function_key(self.name, self.namespace, self.file)

    def __hash__(self) -> int:
        """Return the hash of the node, using its unique key."""
        return hash(self.key())


@dataclass
class ClassNode(TranslatorNode):
    """Store information about a class"""

    # No additional fields needed for now, but can be extended in the future

    def key(self) -> str:
        """Return a unique key for the node, combining namespace and name."""
        return f"{self.module_name()}.{self.name}"


class CallGraphAnalyzer(ast.NodeVisitor):
    def __init__(
        self, language: str, project_path: str = None, files: list[str] = None
    ):
        self.language = language.lower()
        self.project_path = Path(project_path) if project_path else None
        self.files = [Path(f) for f in files] if files else None

        if self.language != "python":
            raise ValueError(
                "Unsupported language: Only Python is supported with ast.NodeVisitor"
            )

        # Initialize attributes for call graph
        self.functions: Dict[str, FunctionNode] = {}
        self.current_namespace: List[str] = []
        self.current_class = None
        self.tree = None  # Store the current AST
        self.code = None
        self.imports = {}
        self.current_file = None
        self.classes: Dict[str, ClassNode] = {}

        self._collect_imports = False
        self._collect_classes = False
        self._collect_functions = False
        self._collect_calls = False

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
            print(f"\nAnalyzing file: {file_path}")
            tree = self.parse_file(str(file_path))
            self.collect_imports(tree)
            # self.collect_classes()
            # self.collect_functions(ast)
            # self.collect_calls(ast)

        import pprint

        print(f"Imported modules: {len(self.imports)}")
        pprint.pprint(self.imports)

    def _is_test_file(self, file_path: Path) -> bool:
        """Determine if a file is a test file based on its name."""
        return file_path.name.startswith("test_") or file_path.name.endswith("_test.py")

    def collect_imports(self, tree: Node):
        self._collect_imports = True
        self.visit(tree)
        self._collect_imports = False

    def collect_classes(self, node: Node):
        self._collect_classes = True
        self.visit(node)
        self._collect_classes = False

    def collect_functions(self, node: Node):
        self._collect_functions = True
        self._maybe_track_namespace(node)
        self.visit(node)
        self._maybe_untrack_namespace(node)
        self._collect_functions = False

    def collect_calls(self, node: Node):
        self._collect_calls = True
        self._maybe_track_namespace(node)
        self.visit(node)
        self._maybe_untrack_namespace(node)
        self._collect_calls = False

    def visit_Import(self, node: ast.Import):
        if not self.collect_imports:
            return
        for alias in node.names:
            self.imports[alias.asname or alias.name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if not self.collect_imports:
            return
        module = node.module or ""
        for alias in node.names:
            self.imports[alias.asname or alias.name] = f"{module}.{alias.name}"
        self.generic_visit(node)

    def _maybe_track_namespace(self, node: Node):
        if node.type in ("function_definition", "class_definition"):
            symbol_name = self._get_symbol_name(self._find_identifier(node))
            self.current_namespace.append(symbol_name)
            if node.type == "class_definition":
                self.current_class = symbol_name

    def _maybe_untrack_namespace(self, node: Node):
        # Legacy - pop namespace when leaving class or function
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

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Process a function definition node."""
        func_name = node.name
        self.current_namespace.append(func_name)

        # FIXME (adam) This is a hack to get the namespace to work for now
        namespace = ".".join(self.current_namespace[:-1])
        function_info = self._get_or_create_function_info(
            node, func_name, namespace=namespace, file=self.current_file
        )
        function_info.start_point = (node.lineno, node.col_offset)
        function_info.end_point = (node.end_lineno, node.end_col_offset)
        self.current_namespace.append(func_name)
        self.generic_visit(node)
        self.current_namespace.pop()

    def _try_resolve_call_with_function_info(
        self, func_name: str, namespace: str | None
    ) -> FunctionNode | None:
        partial_key = (namespace or "") + "." + func_name
        for function_key in self.functions.keys():
            if function_key.endswith(partial_key):
                return self.functions[function_key]
        return None

    def _get_or_create_function_info(
        self,
        node: Node,
        func_name: str,
        namespace: str = None,
        file: str = "UNKNOWN",
    ) -> FunctionNode:
        function_info = self._try_resolve_call_with_function_info(func_name, namespace)
        if function_info:
            if (
                file == "UNKNOWN"
            ):  # This is shorthand for functions calls versus function definitions
                # If we've found the full definition pertaining to that call, use it
                return function_info
            else:
                # Otherwise, we have a function definition and we want to reuse its function_info while amending it
                old_function_key = function_info.key()
                function_info.file = file
                function_info.start_point = node.start_point
                function_info.end_point = node.end_point
                function_info.source_code = self.code[
                    node.start_byte : node.end_byte
                ].decode("utf-8")
                function_info.is_test = self._is_test_function(node)

                # Remove the old function info
                del self.functions[old_function_key]

                # Add the new function info
                self.functions[function_info.key()] = function_info

                return function_info

        # Capture the source code using the node's byte range
        source_code = self.code[node.start_byte : node.end_byte].decode("utf-8")

        function_key = get_function_key(func_name, namespace, file)
        self.functions[function_key] = FunctionNode(
            name=func_name,
            namespace=namespace,
            calls=set(),
            called_by=set(),
            start_point=node.start_point,
            end_point=node.end_point,
            node=node,
            file=file,
            source_code=source_code,  # Store the source code
            is_test=self._is_test_function(node),
        )

        return self.functions[function_key]

    def visit_ClassDef(self, node: ast.ClassDef):
        # Process class definitions
        class_name = node.name
        class_info = ClassNode(name=class_name, node=node, file=self.current_file)
        self.classes[class_info.key()] = class_info
        self.current_namespace.append(class_name)
        self.current_class = class_name
        self.generic_visit(node)
        self.current_namespace.pop()
        self.current_class = None

    def visit_Call(self, node: ast.Call):
        # Process function calls
        if not self.current_namespace:
            return
        current_module = get_module_name(self.current_file)
        caller_key = ".".join([current_module] + self.current_namespace)
        if caller_key not in self.functions:
            return
        callee_info = self._resolve_call(node)
        if isinstance(callee_info, FunctionNode):
            self.functions[caller_key].calls.add(callee_info)
            callee_info.called_by.add(caller_key)
        self.generic_visit(node)

    # def _process_function_call(self, node: Node):
    #     """Process a function call node and update the call graph."""
    #     if not self.current_namespace:
    #         return  # Skip if we're not in any namespace

    #     current_module = get_module_name(self.current_file)
    #     caller_key = ".".join([current_module] + self.current_namespace)
    #     if caller_key not in self.functions:
    #         return  # Skip if we're not in a function

    #     callee_info = self._resolve_call(node)
    #     if isinstance(callee_info, FunctionNode):
    #         self.functions[caller_key].calls.add(callee_info)
    #         callee_info.called_by.add(caller_key)
    #     elif isinstance(callee_info, ClassNode):
    #         self.functions[caller_key].class_deps.add(callee_info.key())

    #         # Add the __init__ function as well for some redundancy, who knows if we'll need it later
    #         function_info = self._get_or_create_function_info(
    #             callee_info.node,
    #             "__init__",
    #             file=callee_info.file,
    #         )
    #         self.functions[caller_key].calls.add(function_info)
    #         function_info.called_by.add(caller_key)

    # def _resolve_call(self, node: Node) -> FunctionNode:
    #     """Resolve the full namespace of a function call."""
    #     if node.type != "call":
    #         return None

    #     # Get the function being called (first child of call node)
    #     func = node.children[0]

    #     if func.type == "identifier":
    #         return self._resolve_simple_call(func)
    #     elif func.type == "attribute":
    #         return self._resolve_attribute_call(func)
    #     elif func.type == "call":
    #         return self._resolve_nested_call(func)

    #     return None

    # def _resolve_simple_call(self, func_node: Node) -> FunctionNode:
    #     """Handle simple function calls like my_function()"""
    #     func_name = self._get_symbol_name(func_node)

    #     # Check if it's a call to an imported module
    #     if func_name in self.imports:
    #         namespace = f"{self.imports[func_name]}"
    #         return self._get_or_create_function_info(
    #             func_node, func_name, namespace=namespace
    #         )

    #     # Check if it's a built-in function
    #     if func_name in dir(builtins):
    #         return self._get_or_create_function_info(
    #             func_node, func_name, file="builtins"
    #         )

    #     # FIXME (adam) Lookup class names in self.classes
    #     # Imperfect way of grabbing the class name for class instantiations
    #     candidate_class_names = [
    #         node_name.split(".")[-2]
    #         for node_name in self.functions.keys()
    #         if len(node_name.split(".")) > 1
    #     ]
    #     if func_name in candidate_class_names:
    #         class_info = self._get_or_create_class_info(func_name, func_node)
    #         return class_info

    #     # Handle module.function() calls
    #     return self._get_or_create_function_info(func_node, func_name)

    def _get_or_create_class_info(
        self, class_name: str, node: Node, file: str = "UNKNOWN"
    ) -> ClassNode:
        """Get or create a ClassInfo object."""
        class_key = f"{self.current_namespace[0]}.{class_name}"
        class_info = self.classes.get(class_key)
        if not class_info:
            # Capture the source code using the node's byte range
            source_code = self.code[node.start_byte : node.end_byte].decode("utf-8")

            class_info = ClassNode(
                name=class_name,
                node=node,
                file=file,
                source_code=source_code,  # Store the source code
            )
            self.classes[class_key] = class_info
        return class_info

    # def _resolve_attribute_call(self, func_node: Node) -> FunctionNode:
    #     """Handle attribute-based calls like obj.method() or module.function()"""
    #     obj = func_node.children[0]
    #     method = func_node.children[2]

    #     obj_name = self._get_symbol_name(obj)
    #     obj_tokens = obj_name.split(".")
    #     method_name = self._get_symbol_name(method)

    #     # Handle calls on imported modules
    #     if obj_name in self.imports:
    #         base_module = self.imports[obj_name]
    #         # FIXME (adam) We're overloading file here with a module name
    #         return self._get_or_create_function_info(
    #             func_node, method_name, file=base_module
    #         )

    #     # Handle self.method() calls
    #     if obj_name == "self" and self.current_class:
    #         namespace = f"{'.'.join(self.current_namespace[:-1])}"
    #         return self._get_or_create_function_info(func_node, method_name, namespace)

    #     # Handle self.instance.method() calls
    #     if obj_name.startswith("self") and self.current_class:
    #         obj_type = self._infer_object_type(obj_name)
    #         if obj_type:
    #             return self._get_or_create_function_info(
    #                 func_node, method_name, namespace=obj_type
    #             )
    #         else:
    #             # NOTE (adam) Default to unknown function that may later be resolved to something.
    #             # This isn't the ideal solution but should do for now
    #             return self._get_or_create_function_info(func_node, method_name)

    #     # Handle class.static_method() calls
    #     if self.current_class and obj_name == self.current_class:
    #         namespace = f"{'.'.join(self.current_namespace[:-1])}"
    #         return self._get_or_create_function_info(
    #             func_node, method_name, namespace=namespace
    #         )

    #     # Handle module.function() calls
    #     if obj_name in self.current_namespace:
    #         return self._get_or_create_function_info(
    #             func_node, method_name, namespace=obj_name
    #         )

    #     # Handle instance.method() calls by trying to determine the object's type
    #     obj_type = self._infer_object_type(obj_name)
    #     if obj_type:
    #         return self._get_or_create_function_info(
    #             func_node, method_name, namespace=obj_type
    #         )

    #     # Fallback: return just obj_name.method_name
    #     return self._get_or_create_function_info(
    #         func_node, method_name, namespace=obj_name
    #     )

    def _infer_object_type(self, obj_name: str) -> str:
        """Attempt to infer the type of an object from the current context.

        This looks for:
        1. Variable assignments like: obj = ClassName()
        2. Function parameters with type hints
        3. Variable annotations

        Returns the fully qualified class name if found, None otherwise.
        """
        # Look for assignment in the current function's scope
        function_key = get_function_key(
            self.current_namespace[0],
            ".".join(self.current_namespace[:-1]) or None,
            self.current_file,
        )
        current_func = self.functions.get(function_key)
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
                        return class_name

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
        self.tree = ast.parse(self.code)  # Store the tree
        return self.tree

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
                for call in sorted(info.calls, key=lambda x: x.key()):
                    output_lines.append(f"    → {call.key()}")
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
            self.collect_imports(ast)
            # self.collect_classes()
            # self.collect_functions(ast)
            # self.collect_calls(ast)

        import pprint

        print(f"Imported modules: {len(self.imports)}")
        pprint.pprint(self.imports)

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
                {node.key() for node in self.get_nodes_at_level(level)}
            )

        # For other levels, find nodes that only call nodes from previous levels
        result = []
        for func in self.functions.values():
            # Skip if already found in previous levels
            if any(func.key() in nodes for nodes in [all_prev_level_nodes]):
                continue

            # Check if all calls are to previous level nodes
            if func.calls and all(
                call.key() in all_prev_level_nodes for call in func.calls
            ):
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
                node_label = f"name={func_info.name}\nnamespace={func_info.namespace}\nfile={func_info.file}"
                dot.node(current_namespace, node_label)
                processed.add(current_namespace)

                # Add edges for each function call
                for called_func in func_info.calls:
                    # Add the called function node if it hasn't been processed
                    if called_func.key() not in processed:
                        called_info = self.functions.get(called_func.key())
                        if called_info:
                            called_label = f"name={called_info.name}\nnamespace={called_info.namespace}\nfile={called_info.file}"
                            dot.node(called_func.key(), called_label)

                    # Add the edge
                    dot.edge(current_namespace, called_func.key())

                    # Add the called function to the stack for further processing
                    stack.append(called_func.key())

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
        query_text = """(
    (import_statement
        (dotted_name) @import.module
    ) @import.statement
)"""

        #         """
        #         query_text = """(
        #     (import_statement
        #         (aliased_import
        #             (dotted_name) @import.module
        #             (as)
        #             (identifier) @import.alias
        #         )
        #     ) @import.statement
        # )"""
        query = self.parser.language.query(query_text)
        captures = query.captures(node)
        module_path = None
        for capture in captures:
            if capture[1] == "import.module":
                module_path = self._get_symbol_name(capture[0])
                alias = module_path  # Default alias is the module path itself
                self.imports[alias] = module_path
            elif capture[1] == "import.alias":
                alias = self._get_symbol_name(capture[0])
                self.imports[alias] = module_path

    def _process_import_from_statement(self, node: Node):
        """Process from-import statements like 'from foo import bar' or 'from foo import bar as baz'."""
        query_text = """
        (
          (import_from_statement
            (relative_import)? @import_from.relative
            (dotted_name)? @import_from.module
            (import
              (aliased_import
                (identifier) @import_from.name
                (as)
                (identifier)? @import_from.alias
              )+
            )
          ) @import_from.statement
        )
        """
        query = self.parser.language.query(query_text)
        captures = query.captures(node)
        module_path = None
        for capture in captures:
            if capture[1] == "import_from.module":
                module_path = self._get_symbol_name(capture[0])
            elif capture[1] == "import_from.name":
                name = self._get_symbol_name(capture[0])
                alias = name  # Default alias is the name itself
                self.imports[alias] = f"{module_path}.{name}"
            elif capture[1] == "import_from.alias":
                alias = self._get_symbol_name(capture[0])
                self.imports[alias] = f"{module_path}.{name}"

    def _process_wildcard_import(self, node: Node):
        """Process wildcard import statements like 'from foo import *'."""
        query_text = """
        (
          (from_import_statement
            (relative_import)? @import_from.relative
            (dotted_name)? @import_from.module
            (wildcard_import) @import_from.star
          ) @import_from.statement
        )
        """
        query = self.parser.language.query(query_text)
        captures = query.captures(node)
        for capture in captures:
            if capture[1] == "import_from.module":
                module_path = self._get_symbol_name(capture[0])
                self.imports[f"{module_path}.*"] = module_path


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

    # analyzer.print_ast(analyzer.tree, 0)
    analyzer.print_call_graph()

    # print("Generating graph...")
    # analyzer.visualize_graph()

    print("Done.")


if __name__ == "__main__":
    main()
