import collections
import json
import logging
import pprint
import re
import subprocess
import sys
from pathlib import Path

import click
from git import Repo

from language_translation.call_graph_analyzer import CallGraphAnalyzer, FunctionNode
from language_translation.code_editor import CodeEditor
from language_translation.file_manager import FileManager
from language_translation.llm_results_parser import LLMResultsParser
from language_translation.llm_translator import LLMTranslator
from language_translation.utils import prompt_user_to_continue


class Translator:

    models = [
        "anthropic/claude-3-sonnet-20240229",
        # "gpt-4o-2024-08-06",
        # "vertex_ai/gemini-1.5-pro-latest",
    ]

    max_execution_validation_retries = 3

    def __init__(self, analyzer: CallGraphAnalyzer):
        self.analyzer = analyzer
        self.file_manager = FileManager(self.analyzer.project_path)
        self.llm_results_parser = LLMResultsParser()
        self.project_path = self.analyzer.project_path
        self.processed_nodes = set()

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

    def _cached_nodes_and_callers(
        self,
    ) -> tuple[list[tuple[FunctionNode, list[FunctionNode]]], list[str], int]:
        cache_path = self._get_cache_path()

        print(f"Found cache file at {cache_path}")
        with open(cache_path) as f:
            cache_data = json.load(f)

        next_index = cache_data.get("next_index", 0)
        next_node = cache_data.get("next_node_name")
        print(f"Resuming translation from node {next_node} (index {next_index})")

        self.processed_nodes = set(cache_data.get("processed_nodes", []))

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

    def _cache_nodes_and_callers(
        self,
        nodes_and_callers: list[tuple[FunctionNode, list[FunctionNode]]],
    ):
        cache_path = self._get_cache_path()

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
            "processed_nodes": list(self.processed_nodes),
        }

        # Write cache to disk
        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)

        print(f"Created new cache file at {cache_path}")
        self.file_manager.commit_all("Committed translation cache")

    def _get_nodes_with_exclusive_callers(
        self,
    ) -> tuple[list[tuple[FunctionNode, list[FunctionNode]]], int]:
        """Get nodes with exclusive callers, using cache if available."""
        cache_path = self._get_cache_path()

        if cache_path.exists():
            return self._cached_nodes_and_callers()

        # If no cache exists, compute the nodes and callers
        nodes_and_callers = self._find_nodes_with_exclusive_callers()

        # Create cache data structure
        self._cache_nodes_and_callers(nodes_and_callers)
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
        cache_data["processed_nodes"] = list(self.processed_nodes)

        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)

    def _find_nodes_with_exclusive_callers(
        self,
    ) -> list[tuple[FunctionNode, list[FunctionNode]]]:

        if not self.analyzer.get_leaf_nodes():
            self.analyzer.analyze()

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
                # FIXME (adam) NOTE that this will result in duplicate deps since we're not checking whether they're exclusive
                # or whether they've been added before
                for dep in node.deps:
                    # print(f"  Piggybacking dep: {dep.name}")
                    nodes_and_exclusive_callers.append((dep, [node]))

                nodes_and_exclusive_callers.append((node, exclusive_callers))
        return nodes_and_exclusive_callers

    def setup_tree_and_translate(
        self,
        llm_translator: LLMTranslator,
        nodes_to_translate: list[FunctionNode],
    ) -> dict[str, str]:
        code_snippets_by_file = collections.defaultdict(list)
        for node in nodes_to_translate:
            code_snippets_by_file[node.file].append(node.source_code)

        translated_go_code = llm_translator.translate(
            code_snippets_by_file,
            nodes_to_translate[0].name,
        )

        print("\nTranslated Go code:")
        print(translated_go_code)

        return self.llm_results_parser.parse_translations(translated_go_code)

    def retry_translate(
        self,
        llm_translator: LLMTranslator,
        last_test_output: str,
        node_name: str,
    ) -> dict[str, str]:
        translated_go_code = llm_translator.retry(last_test_output, node_name)
        return self.llm_results_parser.parse_translations(translated_go_code)

    def retry_insertion(
        self,
        code_editor: CodeEditor,
        last_test_output: str,
        node_name: str,
    ) -> dict[str, str]:
        translated_go_code = code_editor.retry(last_test_output, node_name)

        print("\nNew Go file from retry:")
        print(translated_go_code)

        return self.llm_results_parser.parse_translations(translated_go_code)

    def test_insertions(self, insertions: dict[str, str]) -> tuple[bool, str, str]:
        test_files = []
        for rel_fname, new_code in insertions.items():
            full_path = Path(rel_fname)
            sandbox_path = (
                self.file_manager.go_project_path / "sandbox" / full_path.name
            )
            print(f"Full path name: {full_path.name}")
            if full_path.name.endswith("_test.go"):
                test_files.append(full_path)
            # Rewrite package name to sandbox
            new_code = re.sub(r"^package \w+", "package sandbox", new_code, count=1)
            self.file_manager.rewrite_file(sandbox_path, new_code)

        test_files = ["sandbox/" + test_file.name for test_file in test_files]

        command = ["go", "test", *test_files]
        print(f"Running command: {' '.join(command)}")  # Print the command
        result = subprocess.run(
            command,
            cwd=self.file_manager.get_target_repo_path(),
            capture_output=True,
            text=True,
        )

        print(f"\n--- SANDBOX TEST RESULTS ---")
        print(f"STDOUT:")
        print(result.stdout)
        print(f"STDERR:")
        print(result.stderr)
        print(f"--- END SANDBOX TEST RESULTS ---")

        return result.returncode == 0, result.stdout, result.stderr

    def test_repo(self) -> tuple[bool, str, str]:
        """Run tests in the target repository."""
        repo_path = self.file_manager.get_target_repo_path()

        # Run go test on the changed files
        result = subprocess.run(
            ["go", "test", "./..."],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )

        print(f"\n--- REPO TEST RESULTS ---")
        print(f"STDOUT:")
        print(result.stdout)
        print(f"STDERR:")
        print(result.stderr)
        print(f"--- END REPO TEST RESULTS ---")

        return result.returncode == 0, result.stdout, result.stderr

    def translate_tree2(self, nodes_to_translate: list[FunctionNode]) -> None:
        """Translate code and retry until tests pass, retrying and failing over to different models until successful."""
        self.file_manager.reset_git()

        node_name = nodes_to_translate[0].name

        # Single LLM translator pertains to a single conversation with a single LLM, so if we failover to another model
        # we start the conversation over and thereby need to create a new LLM translator instance
        tests_passed = False
        model = self.models[0]
        for model in self.models:  # Loop over the models that LLMTranslator will use

            if tests_passed:
                break

            print(f"Using model: {model}")
            llm_translator = LLMTranslator(model, self.file_manager)

            # TODO (adam) Put this in a retry loop
            insertions = self.setup_tree_and_translate(
                llm_translator, nodes_to_translate
            )
            prompt_user_to_continue(
                "Initial node translation complete. Press y to run sandbox tests."
            )

            tree_tests_passed, _, stderr = self.test_insertions(insertions)

            attempt = 0
            while (
                not tree_tests_passed
                and attempt < self.max_execution_validation_retries
            ):
                retry_insertions = self.retry_translate(
                    llm_translator, stderr, node_name
                )
                prompt_user_to_continue(
                    "Retry translation complete. Press y to run sandbox tests."
                )

                tree_tests_passed, _, _ = self.test_insertions(retry_insertions)

                if tree_tests_passed:
                    prompt_user_to_continue(
                        "Sandbox tests passed. Press y to run code insertion completions."
                    )
                    break

                attempt += 1

            if attempt >= self.max_execution_validation_retries:
                print(
                    "Failed to generate passing translation after trying all models. Please repair and "
                    "commit unstaged code in the target repository, and restart translation."
                )
                sys.exit(1)

            attempt = 0
            for model in self.models:  # Loop over the models that CodeEditor will use
                if tests_passed:
                    break
                code_editor = CodeEditor(model, self.file_manager)
                code_editor.insert_code(insertions, node_name)
                prompt_user_to_continue(
                    "Code insertion complete. Press y to run repo tests."
                )

                tests_passed, _, stderr = self.test_repo()

                while (
                    not tests_passed and attempt < self.max_execution_validation_retries
                ):
                    prompt_user_to_continue(
                        "Repo tests complete. Press y to continue to retry code insertion."
                    )
                    insertions = self.retry_insertion(code_editor, stderr, node_name)
                    prompt_user_to_continue(
                        "Code insertion complete. Press y to run repo tests."
                    )

                    tests_passed, stdout, stderr = self.test_repo()
                    prompt_user_to_continue(
                        "Repo tests complete. Press y to continue to next node or retry code insertion."
                    )

                    attempt += 1

        if not tests_passed:
            print(
                "Failed to generate passing translation after trying all models. Please repair and "
                "commit unstaged code in the target repository, and restart translation."
            )
            sys.exit(1)

    def translate(self) -> str:
        self.file_manager.setup_project()

        nodes_and_callers, start_index = self._get_nodes_with_exclusive_callers()
        print(f"Found {len(nodes_and_callers)} nodes to translate")
        if start_index > 0:
            print(
                f"Starting from node at index {start_index}: {nodes_and_callers[start_index][0].name}"
            )

        for i, (node, exclusive_callers) in enumerate(
            nodes_and_callers[start_index:], start=start_index
        ):

            print(f"\n===============================")
            print(f"Translating node {i+1}/{len(nodes_and_callers)}: {node.name}")

            # We set up files a subtree at a time
            nodes_to_translate = [node] + exclusive_callers
            nodes_to_translate = [
                n for n in nodes_to_translate if n.key() not in self.processed_nodes
            ]

            if not nodes_to_translate:
                print(f"All nodes have already been translated: {node.name}")
                continue

            py_files = set([Path(f.file) for f in nodes_to_translate])
            self.file_manager.setup_files(py_files)
            self.file_manager.setup_sandbox_files(py_files)

            self.translate_tree2(nodes_to_translate)
            self._update_cache_index(i + 1, nodes_and_callers)
            self.file_manager.commit_all(f"Translated node: {node.name}")

            self.processed_nodes.update(node.key() for node in nodes_to_translate)


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

    logging.basicConfig(level=logging.INFO)

    analyzer = CallGraphAnalyzer(
        language=language,
        project_path=project_path if project_path else None,
        files=files if files else None,
    )
    # analyzer.analyze()

    translator = Translator(analyzer)
    translator.translate()

    # analyzer.print_call_graph()

    # print("Generating graph...")
    # analyzer.visualize_graph()

    print("Done.")


if __name__ == "__main__":
    main()
