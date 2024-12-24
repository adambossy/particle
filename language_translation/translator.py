from pathlib import Path
from typing import Dict

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


class Translator:
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

    def analyze(self):
        """Analyze either the project directory or specific files."""
        if self.project_path:
            self.analyze_project()
        elif self.files:
            self.analyze_files()
        else:
            raise ValueError("Either project_path or files must be provided")

    def analyze_files(self):
        """Analyze specific files."""
        for file_path in self.files:
            print(f"\nAnalyzing file: {file_path}")
            ast = self.parse_file(str(file_path))
            self.print_ast(ast)

    def parse_file(self, file_path: str) -> Node:
        """Parse a single file and return its AST."""
        with open(file_path, "rb") as f:
            code = f.read()
        tree = self.parser.parse(code)
        return tree.root_node

    def print_ast(self, node: Node, level: int = 0):
        """Recursively print the AST in a readable format."""
        node_info = self._format_node_info(node)
        indent = self._get_indent(level)
        print(f"{indent}{node_info}")

        self._print_children(node, level)

    def _format_node_info(self, node: Node) -> str:
        """Format the node information including type, field name, and text content."""
        node_info = self._get_node_type(node)
        node_info = self._add_field_name(node, node_info)
        node_info = self._add_node_text(node, node_info)
        return node_info

    def _get_node_type(self, node: Node) -> str:
        """Get the basic node type."""
        return f"{node.type}"

    def _add_field_name(self, node: Node, node_info: str) -> str:
        """Add field name to node info if it exists."""
        # Check if the node has a field name using the correct method or attribute
        field_name = getattr(node, "field_name", None)
        if field_name:
            return f"{field_name}: {node_info}"
        return node_info

    def _add_node_text(self, node: Node, node_info: str) -> str:
        """Add node text content if it exists and is not empty."""
        text = node.text.decode("utf-8") if node.text else ""
        if text and len(text.strip()) > 0:
            return f"{node_info} [{text.strip()}]"
        return node_info

    def _get_indent(self, level: int) -> str:
        """Generate the appropriate indentation for the current level."""
        return "  " * level

    def _print_children(self, node: Node, level: int):
        """Print all children nodes recursively."""
        for child in node.children:
            self.print_ast(child, level + 1)

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
    """Analyze Python or Swift code and print the AST."""
    if not project_path and not files:
        raise click.UsageError("Either --project-path or --files must be provided")
    if project_path and files:
        raise click.UsageError("Cannot use both --project-path and --files together")

    translator = Translator(
        language=language,
        project_path=project_path if project_path else None,
        files=files if files else None,
    )
    translator.analyze()


if __name__ == "__main__":
    main()
