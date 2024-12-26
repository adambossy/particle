from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

import click
import tree_sitter_python
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
        self.current_class = None  # Track current class during traversal
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
            print(f"\nAnalyzing file: {file_path}")
            self.current_namespace = [Path(file_path).stem]  # Start with module name
            ast = self.parse_file(str(file_path))
            self.build_call_graph(ast)

    def build_call_graph(self, node: Node):
        """Build the call graph by traversing the AST."""
        if node.type == "function_definition":
            self._process_function_definition(node)
        elif node.type == "class_definition":
            self._process_class_definition(node)
        elif node.type == "call":
            self._process_function_call(node)

        for child in node.children:
            self.build_call_graph(child)

        # Pop namespace when leaving class or function
        if node.type in ("function_definition", "class_definition"):
            self.current_namespace.pop()
            if node.type == "class_definition":
                self.current_class = None

    def _process_function_definition(self, node: Node):
        """Process a function definition node."""
        func_name = self._get_symbol_name(self._find_identifier(node))
        self.current_namespace.append(func_name)

        full_name = ".".join(self.current_namespace)
        self.functions[full_name] = FunctionInfo(
            name=func_name,
            namespace=full_name,
            calls=set(),
            called_by=set(),
            start_point=node.start_point,
            end_point=node.end_point,
            node=node,
        )

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

        callee = self._resolve_call(node)
        if callee:
            self.functions[caller_namespace].calls.add(callee)
            if callee in self.functions:
                self.functions[callee].called_by.add(caller_namespace)

    def _resolve_call(self, node: Node) -> str:
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

    def _resolve_simple_call(self, func_node: Node) -> str:
        """Handle simple function calls like my_function()"""
        func_name = self._get_symbol_name(func_node)

        # Check if it's a built-in function
        if hasattr(__builtins__, func_name):
            return f"builtins.{func_name}"

        # Return full namespace path
        return f"{'.'.join(self.current_namespace[:-1])}.{func_name}"

    def _resolve_attribute_call(self, func_node: Node) -> str:
        """Handle attribute-based calls like obj.method() or module.function()"""
        obj = func_node.children[0]
        method = func_node.children[2]

        obj_name = self._get_symbol_name(obj)
        method_name = self._get_symbol_name(method)

        # Handle self.method() calls
        if obj_name == "self" and self.current_class:
            return f"{'.'.join(self.current_namespace[:-1])}.{method_name}"

        # Handle class.static_method() calls
        if self.current_class and obj_name == self.current_class:
            return f"{'.'.join(self.current_namespace[:-1])}.{method_name}"

        # Handle module.function() calls
        if obj_name in self.current_namespace:
            return f"{obj_name}.{method_name}"

        # Handle instance.method() calls
        # This is a basic implementation - could be enhanced with type inference
        return f"{obj_name}.{method_name}"

    def _resolve_nested_call(self, func_node: Node) -> str:
        """Handle nested calls like get_object().method()"""
        # For nested calls, we focus on the final method being called
        # This is a simplified implementation
        if func_node.children and func_node.children[-1].type == "attribute":
            return self._resolve_attribute_call(func_node.children[-1])
        return None

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
            self.print_ast(ast)


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
    analyzer.print_call_graph()


if __name__ == "__main__":
    main()
