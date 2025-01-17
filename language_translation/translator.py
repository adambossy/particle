import collections
import json
import pprint
import subprocess
from pathlib import Path

import click
from git import Repo

from language_translation.call_graph_analyzer import CallGraphAnalyzer, FunctionNode
from language_translation.code_editor import CodeEditor
from language_translation.file_manager import FileManager
from language_translation.llm_results_parser import LLMResultsParser
from language_translation.llm_translator import LLMTranslator


class Translator:

    def __init__(self, analyzer: CallGraphAnalyzer):
        self.analyzer = analyzer
        self.file_manager = FileManager(self.analyzer.project_path)
        self.llm_translator = LLMTranslator(self.file_manager)
        self.llm_results_parser = LLMResultsParser()
        self.code_editor = CodeEditor(self.file_manager)
        self.project_path = self.analyzer.project_path

    def _get_git_sha(self) -> str:
        """Get the latest git SHA of the project."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()[:8]  # Return first 8 chars of the SHA
        except subprocess.CalledProcessError as e:
            print(f"Warning: Could not get git SHA: {e}")
            return "no_git"

    def _get_cache_path(self) -> Path:
        """Get the path to the translation cache file."""
        git_sha = self._get_git_sha()
        target_repo_path = (
            self.file_manager.get_target_repo_path()
        )  # Assuming file_manager is an instance of FileManager
        return Path(target_repo_path) / f"translation_cache_{git_sha}.json"

    def _serialize_node(self, node: FunctionNode) -> dict:
        """Serialize a FunctionNode to a JSON-compatible dictionary."""
        return {
            "name": node.name,
            "file": node.file,
            # "namespace": node.namespace,
            # "lineno": node.lineno,
            # "end_lineno": node.end_lineno,
            # "class_deps": list(node.class_deps),
            # "var_deps": list(node.var_deps),
            "source_code": node.source_code,
            # "scope": node.scope.name if node.scope else None,
            # Add other attributes as needed
        }

    def _deserialize_node(self, data: dict) -> FunctionNode:
        """Deserialize a dictionary back to a FunctionNode."""
        # Reconstruct the FunctionNode from the dictionary
        # Note: You may need to handle the scope and other complex attributes separately
        return FunctionNode(
            name=data["name"],
            file=data["file"],
            # namespace=data["namespace"],
            # lineno=data["lineno"],
            # end_lineno=data["end_lineno"],
            # class_deps=set(data["class_deps"]),
            # var_deps=set(data["var_deps"]),
            source_code=data["source_code"],
            # You will need to handle the scope reconstruction
        )

    def _get_nodes_with_exclusive_callers(
        self,
    ) -> tuple[list[tuple[FunctionNode, list[FunctionNode]]], int]:
        """Get nodes with exclusive callers, using cache if available."""
        cache_path = self._get_cache_path()

        if cache_path.exists():
            print(f"Found cache file at {cache_path}")
            with open(cache_path) as f:
                cache_data = json.load(f)

            next_index = cache_data.get("next_index", 0)
            next_node = cache_data.get("next_node_name")
            print(f"Resuming translation from node {next_node} (index {next_index})")

            # Deserialize nodes and exclusive callers
            nodes_and_callers = [
                (
                    self._deserialize_node(node_data["node"]),
                    [
                        self._deserialize_node(caller)
                        for caller in node_data["exclusive_callers"]
                    ],
                )
                for node_data in cache_data["nodes_and_callers"]
            ]

            return nodes_and_callers, next_index

        # If no cache exists, compute the nodes and callers
        nodes_and_callers = self._find_nodes_with_exclusive_callers()

        # Create cache data structure
        cache_data = {
            "next_index": 0,
            "next_node_name": (
                nodes_and_callers[0][0].name if nodes_and_callers else None
            ),
            "nodes_and_callers": [
                {
                    "node": self._serialize_node(node),
                    "exclusive_callers": [
                        self._serialize_node(caller) for caller in callers
                    ],
                }
                for node, callers in nodes_and_callers
            ],
        }

        # Write cache to disk
        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)

        print(f"Created new cache file at {cache_path}")
        return nodes_and_callers, 0

    def _update_cache_index(self, index: int, nodes_and_callers: list):
        """Update the current index in the cache file."""
        cache_path = self._get_cache_path()

        with open(cache_path) as f:
            cache_data = json.load(f)

        cache_data["next_index"] = index
        cache_data["next_node_name"] = (
            nodes_and_callers[index][0].name if index < len(nodes_and_callers) else None
        )

        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)

    def _find_nodes_with_exclusive_callers(
        self,
    ) -> list[tuple[FunctionNode, list[FunctionNode]]]:
        nodes_and_exclusive_callers = []
        for node in self.analyzer.get_leaf_nodes():
            if node.is_test() or not (
                node.scope.parent and node.scope.parent.is_module()
            ):
                continue
            # Callers that exclusively call this function and no others. In the future, we can get more sophisticated
            # and get callers that only call functions at the same level as this one, but we're starting simple
            exclusive_callers = [
                caller
                for caller in node.called_by
                if len(caller.calls) == 1 and caller.is_test()
            ]
            if exclusive_callers:
                print(f"Translating {node.name}")
                print(f"  File: {node.file}")
                print(
                    f"  Scope: {node.scope} module_level? {node.scope.parent and node.scope.parent.is_module()}"
                )
                nodes_and_exclusive_callers.append((node, exclusive_callers))
            for caller in exclusive_callers:
                print(f"  Exclusive caller: {caller.name}")
        return nodes_and_exclusive_callers

    def _translate_tree(
        self,
        node: FunctionNode,
        exclusive_callers: list[FunctionNode],
        py_files: list[Path],
    ) -> dict[str, str]:
        code_snippets_by_file = collections.defaultdict(list)
        code_snippets_by_file[node.file].append(node.source_code)

        for caller in exclusive_callers:
            code_snippets_by_file[caller.file].append(caller.source_code)

        translated_go_code = self.llm_translator.translate(code_snippets_by_file)
        print("\nTranslated Go code:")
        print(translated_go_code)

        target_filenames = set(
            self.file_manager.get_target_file_path(py_file).as_posix()
            for py_file in py_files
        )
        print(f"\nTarget files: {target_filenames}")

        return self.llm_results_parser.parse_translations(
            translated_go_code, target_filenames
        )

    def _run_tests(self):
        """Run tests in the target repository."""
        repo_path = self.file_manager.get_target_repo_path()

        # Run go test on the changed files
        result = subprocess.run(
            ["go", "test", "./..."],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        print("\nTest Results:")
        print(result.stdout)

        # Check if any tests failed
        if result.returncode != 0:
            print("\nTests failed - stopping translation")
            print("Error output:")
            print(result.stderr)
            raise RuntimeError("Tests failed after translation")

    def translate(self) -> str:
        self.file_manager.setup_project()

        nodes_and_callers, start_index = self._get_nodes_with_exclusive_callers()
        print(f"Found {len(nodes_and_callers)} nodes to translate")

        for i, (node, exclusive_callers) in enumerate(
            nodes_and_callers[start_index:], start=start_index
        ):
            print(f"\nTranslating node {i+1}/{len(nodes_and_callers)}: {node.name}")

            # We set up files a subtree at a time
            py_files = set(
                [Path(f.file) for f in exclusive_callers] + [Path(node.file)]
            )
            self.file_manager.setup_files(py_files)
            edits = self._translate_tree(node, exclusive_callers, py_files)

            print(f"Edits ({len(edits)}): {edits.keys()}")
            self.code_editor.apply_edits(edits)

            # Run tests before updating cache
            self._run_tests()

            # Update cache with next index
            self._update_cache_index(i + 1, nodes_and_callers)

            # Commit the changes
            self._commit_changes(node.name)

    def _commit_changes(self, node_name: str):
        """Commit changes to the git repository."""
        repo_path = self.file_manager.get_target_repo_path()
        repo = Repo(repo_path)

        # Stage all changes
        repo.git.add(A=True)

        # Commit with a message including the node name
        commit_message = f"Translated node: {node_name}"
        repo.index.commit(commit_message)

        print(f"Committed changes for node: {node_name}")


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

    translator = Translator(analyzer)
    translator.translate()

    # analyzer.print_call_graph()

    # print("Generating graph...")
    # analyzer.visualize_graph()

    print("Done.")


if __name__ == "__main__":
    main()
