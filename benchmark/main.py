import abc
import asyncio
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import click

from particle.file_manager import FileManager
from particle.llm_translator import LLMTranslator

SUPPORTED_MODELS = [
    "gpt-4o-2024-08-06",
    "gpt-3.5-turbo",
    "anthropic/claude-3-5-sonnet-20241022",
    "vertex_ai/gemini-1.5-pro-latest",
    "deepseek-coder",
]

SUPPORTED_LANGUAGES = [
    "python",
    "go",
    "typescript",
    "rust",
    "java",
    "cpp",
]


@dataclass
class CodeSample:
    source_code: str
    source_path: str
    source_interface: str
    source_test_code: str
    source_test_path: str
    target_code: str | None = None
    target_path: str | None = None
    target_interface: str | None = None
    target_test_code: str | None = None
    target_test_path: str | None = None

    # NOTE (adam) Exercism-specific test code that's auto-generated into a separate file for each
    # exercise
    cases_test_code: str | None = None
    cases_test_path: str | None = None


class SampleCollector(abc.ABC):
    @abc.abstractmethod
    def get_code_samples(self) -> Generator[CodeSample, None, None]:
        """Returns a generator of CodeSample instances."""
        pass


class ExercismSampleCollector(SampleCollector):
    REPO_URL_TEMPLATE = "https://github.com/exercism/{}"

    # File extension mappings for different languages
    LANGUAGE_EXTENSIONS = {
        "python": "py",
        "go": "go",
        "typescript": "ts",
        "rust": "rs",
        "java": "java",
        "cpp": "cpp",
    }

    def __init__(self, source_lang: str, target_lang: str, workspace_dir: Path):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.workspace_dir = workspace_dir
        self.source_repo_path = workspace_dir / self.source_lang
        self.target_repo_path = workspace_dir / self.target_lang

        # Clone repositories
        self._clone_repos()

    def _clone_repos(self) -> None:
        """Clone the source and target language repositories."""
        for lang in [self.source_lang, self.target_lang]:
            repo_url = self.REPO_URL_TEMPLATE.format(lang)
            repo_path = self.workspace_dir / lang

            print(f"Cloning {repo_url} into {repo_path}")

            if not repo_path.exists():
                subprocess.run(
                    ["git", "clone", "--depth=1", repo_url, str(repo_path)],
                    check=True,
                    capture_output=True,
                )

    def _get_exercise_pairs(self) -> Generator[tuple[Path, Path], None, None]:
        """Find matching exercise pairs between source and target repos."""
        source_exercises = self.source_repo_path / "exercises" / "practice"
        target_exercises = self.target_repo_path / "exercises" / "practice"

        # Get all exercise directories in source repo
        for source_ex_path in source_exercises.glob("*"):
            if not source_ex_path.is_dir():
                continue

            exercise_name = source_ex_path.name
            target_ex_path = target_exercises / exercise_name

            # Check if the same exercise exists in target repo
            if target_ex_path.exists():
                yield source_ex_path, target_ex_path

    def get_code_samples(
        self, num_samples: int = None
    ) -> Generator[CodeSample, None, None]:
        """Generate CodeSample instances for matching exercises."""
        source_ext = self.LANGUAGE_EXTENSIONS[self.source_lang]
        target_ext = self.LANGUAGE_EXTENSIONS[self.target_lang]

        samples_yielded = 0

        for source_ex_path, target_ex_path in self._get_exercise_pairs():
            if num_samples is not None and samples_yielded >= num_samples:
                break

            exercise_name = source_ex_path.name

            # Contains solution code to translate
            source_file = source_ex_path / ".meta" / f"example.{source_ext}"
            if not source_file.exists():
                continue

            # Contains interface stub
            source_interface = source_ex_path / f"{exercise_name}.{source_ext}"
            if not source_interface.exists():
                continue

            # Contains tests to help translation along
            source_test_file = source_ex_path / f"{exercise_name}_test.{source_ext}"
            if not source_test_file.exists():
                continue

            # Contains interface stub
            target_code_file = target_ex_path / f"{exercise_name}.{target_ext}"
            if not target_code_file.exists():
                continue

            # Contains tests that translated code must pass
            target_test_file = target_ex_path / f"{exercise_name}_test.{target_ext}"
            if not target_test_file.exists():
                continue

            # Contains tests that translated code must pass
            cases_test_file = target_ex_path / f"cases_test.{target_ext}"
            if not cases_test_file.exists():
                continue

            try:
                # Read source and test files
                with open(source_file) as f:
                    source_code = f.read()
                with open(source_interface) as f:
                    source_interface = f.read()
                with open(source_test_file) as f:
                    source_test_code = f.read()
                with open(target_code_file) as f:
                    target_interface = f.read()
                with open(target_test_file) as f:
                    target_test_code = f.read()
                with open(cases_test_file) as f:
                    cases_test_code = f.read()

                # Create relative paths from workspace directory
                source_rel_path = source_file.relative_to(self.source_repo_path)
                source_test_rel_path = source_test_file.relative_to(
                    self.source_repo_path
                )
                test_rel_path = target_test_file.relative_to(self.target_repo_path)
                cases_test_rel_path = cases_test_file.relative_to(self.target_repo_path)

                yield CodeSample(
                    source_code=source_code,
                    source_path=str(source_rel_path),
                    source_interface=source_interface,
                    source_test_code=source_test_code,
                    source_test_path=str(source_test_rel_path),
                    target_interface=target_interface,
                    target_test_code=target_test_code,
                    target_test_path=str(test_rel_path),
                    cases_test_code=cases_test_code,
                    cases_test_path=str(cases_test_rel_path),
                )
                samples_yielded += 1

            except (IOError, OSError) as e:
                print(f"Error processing {exercise_name}: {e}")
                continue


class Sandbox:
    """Manages a sandbox environment for test running logic."""

    def __init__(self, workspace_dir: Path, target_lang: str):
        self.workspace_dir = workspace_dir
        self.target_lang = target_lang
        self.sandbox_path = workspace_dir / "sandbox"
        self.setup_sandbox()

    def setup_sandbox(self) -> None:
        """Create the sandbox directory if it doesn't exist."""
        self.sandbox_path.mkdir(parents=True, exist_ok=True)

    def setup_sandbox_files(self, *files: Path | str) -> None:
        """Set up empty files in the sandbox directory.

        Args:
            *files: Variable number of file paths to create in sandbox
        """
        for file_path in files:
            if isinstance(file_path, str):
                file_path = Path(file_path)

            sandbox_file = self.sandbox_path / file_path.name
            sandbox_file.touch(exist_ok=True)

    def create_code_file(self, rel_path: Path | str, source_code: str) -> Path:
        """Create a file containing the translated code in the sandbox.

        Args:
            rel_path: Relative path for the target file
            source_code: Code content to write to the file

        Returns:
            Path to the created file
        """
        if isinstance(rel_path, str):
            rel_path = Path(rel_path)

        file_path = self.sandbox_path / rel_path.name
        file_path.write_text(source_code)
        return file_path

    def create_test_file(self, rel_path: Path | str, test_code: str) -> Path:
        """Create a test file in the sandbox.

        Args:
            rel_path: Relative path for the test file
            test_code: Test code content to write to the file

        Returns:
            Path to the created test file
        """
        if isinstance(rel_path, str):
            rel_path = Path(rel_path)

        file_path = self.sandbox_path / rel_path.name
        file_path.write_text(test_code)
        return file_path

    # NOTE (adam) This function is Exercism-specific
    def create_cases_test_file(self, rel_path: Path | str, test_code: str) -> Path:
        if isinstance(rel_path, str):
            rel_path = Path(rel_path)

        file_path = self.sandbox_path / rel_path.name
        file_path.write_text(test_code)
        return file_path

    def get_test_command(self, test_file: Path) -> tuple[list[str], dict]:
        """Returns the command to run tests and any necessary environment setup."""
        test_dir = test_file.parent

        commands = {
            "python": (
                ["python", "-m", "pytest", str(test_file)],
                {"env": {**os.environ, "PYTHONPATH": str(test_dir)}},
            ),
            "go": (
                ["go", "test", "./..."],
                {"cwd": test_dir},
            ),
            "typescript": (
                ["npm", "test"],
                {"cwd": test_dir},
            ),
            "rust": (
                ["cargo", "test"],
                {"cwd": test_dir},
            ),
            "java": (
                ["./gradlew", "test"],
                {"cwd": test_dir},
            ),
            "cpp": (
                ["cmake", "--build", "build", "--target", "test"],
                {"cwd": test_dir},
            ),
        }

        return commands.get(self.target_lang, ([], {}))

    def setup_test_environment(self, test_dir: Path) -> None:
        """Set up any necessary test environment for the given language."""
        if self.target_lang == "python":
            # Create requirements.txt if needed
            requirements = test_dir / "requirements.txt"
            if not requirements.exists():
                with open(requirements, "w") as f:
                    f.write("pytest\n")
            subprocess.run(["pip", "install", "-r", str(requirements)], check=True)

        elif self.target_lang == "go":
            print(f"Setting up go test environment for directory {test_dir}")
            # Initialize go module if needed
            if not (test_dir / "go.mod").exists():
                module_name = f"exercism/{test_dir.name}"
                subprocess.run(
                    ["go", "mod", "init", module_name], cwd=test_dir, check=True
                )
                # Download test dependencies
                subprocess.run(["go", "mod", "tidy"], cwd=test_dir, check=True)

        elif self.target_lang in ["typescript", "rust", "java", "cpp"]:
            raise NotImplementedError(
                f"Test environment setup for {self.target_lang} is not yet implemented"
            )

        else:
            raise ValueError(f"Unsupported language: {self.target_lang}")

    def run_tests(self, test_file: Path) -> subprocess.CompletedProcess:
        """Run tests for the given language and return the result."""
        # Set up test environment
        test_dir = test_file.parent
        self.setup_test_environment(test_dir)

        # Get test command for target language
        cmd, run_kwargs = self.get_test_command(test_file)
        if not cmd:
            raise ValueError(f"Unsupported target language: {self.target_lang}")

        # Run tests
        print("\n--- RUNNING TESTS ---")
        print("  CWD:", run_kwargs["cwd"])
        print("  CMD:", " ".join(cmd))

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=run_kwargs["cwd"]
        )

        result.stdout = result.stdout or ""
        result.stderr = result.stderr or ""

        print("Return code:", result.returncode)
        print(result.stdout)
        print(result.stderr)
        print("--- END RUNNING TESTS ---")

        return result


async def process_sample(
    sample: CodeSample,
    workspace_dir: Path,
    file_manager: FileManager,
    sandbox: Sandbox,
    model: str,
    source_lang: str,
    target_lang: str,
    results_dir: Path,
) -> None:
    """Process a single code sample."""
    # Create necessary directories
    # TODO (adam) Have Sandbox manage the portion before the try/except block
    source_file = workspace_dir / sample.source_path
    test_file = workspace_dir / "sandbox" / Path(sample.target_test_path).name
    cases_test_file = workspace_dir / "sandbox" / Path(sample.cases_test_path).name
    translated_file = (
        workspace_dir
        / "sandbox"
        / f"{test_file.stem.replace('_test', '')}.{ExercismSampleCollector.LANGUAGE_EXTENSIONS[target_lang]}"
    )

    print(f"Source file: {source_file}")
    print(f"Test file: {test_file}")

    print(f"\nSandbox:")
    print(f"Translated file: {translated_file}")
    print(f"Copied test file: {test_file}")

    os.makedirs(os.path.dirname(source_file), exist_ok=True)
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    os.makedirs(os.path.dirname(cases_test_file), exist_ok=True)

    # Write source and test files
    with open(source_file, "w") as f:
        f.write(sample.source_code)
    with open(test_file, "w") as f:
        f.write(sample.target_test_code)
    with open(cases_test_file, "w") as f:
        f.write(sample.cases_test_code)

    try:
        # Initialize translator and translate code
        translator = LLMTranslator(model, file_manager)

        # Create a dictionary mapping source file to code snippets
        code_snippets = {sample.source_path: [sample.source_code]}

        # NOTE (adam) Exercism-specific instructions since the repos are maintained separately and
        # so the function names differ
        special_instructions = f"""
The following code contains the interface that must be implemented to satisfy tests in the target language:
{sample.source_interface}

The following code contains the interface that must be implemented to satisfy tests in the target language:
{sample.target_interface}

Ensure that the function name in the source code gets translated using the function name in the target language.
"""

        # Translate the code
        translated_code = translator.translate(
            code_snippets,
            special_instructions=special_instructions,
        )

        # Write translated code
        print(f"Writing translated code to {translated_file}")
        with open(translated_file, "w") as f:
            f.write(translated_code)

        # Run tests using TestRunner
        result = sandbox.run_tests(test_file)
        print(f"Test result: {result}")
        print(f"Test Output (stdout): {result.stdout}")
        print(f"Test Output (stderr): {result.stderr}")

        # Create results file
        exercise_name = source_file.parent.name
        result_file = results_dir / f"{exercise_name}_results.txt"

        with open(result_file, "w") as f:
            f.write(f"Exercise: {exercise_name}\n")
            f.write(f"Source Language: {source_lang}\n")
            f.write(f"Target Language: {target_lang}\n")
            f.write(f"Model: {model}\n")
            f.write(f"Test Return Code: {result.returncode}\n\n")

            num_attempts = 0

            while result.returncode != 0 and num_attempts < 10:
                f.write("Test Output (stderr):\n")
                f.write(result.stderr)
                f.write("\nTest Output (stdout):\n")
                f.write(result.stdout)

                # If tests failed, try to translate again with the error output
                last_test_output = result.stderr + "\n" + result.stdout
                translated_code = translator.retry(
                    last_test_output,
                    sample.target_test_code,
                )

                # Write the retried translation
                with open(translated_file, "w") as f_retry:
                    f_retry.write(translated_code)

                # Run tests again
                result = sandbox.run_tests(test_file)

                f.write("\n\nRetry attempt:\n")
                f.write(f"Test Return Code: {result.returncode}\n")
                if result.returncode != 0:
                    f.write("Test Output (stderr):\n")
                    f.write(result.stderr)
                    f.write("\nTest Output (stdout):\n")
                    f.write(result.stdout)
                else:
                    f.write("Tests passed successfully after retry!\n")
                    print("Tests passed successfully after retry!")

                num_attempts += 1

            if result.returncode == 0:
                f.write("Tests passed successfully!\n")
                print("Tests passed successfully!")

    except subprocess.CalledProcessError as e:
        # Handle setup failures
        error_file = results_dir / f"{exercise_name}_error.txt"
        with open(error_file, "w") as f:
            f.write(f"Error setting up/running tests: {str(e)}\n")
            if e.stdout:
                f.write(f"\nStdout:\n{e.stdout.decode()}")
            if e.stderr:
                f.write(f"\nStderr:\n{e.stderr.decode()}")


async def evaluate(
    model: str, source_lang: str, target_lang: str, num_samples: int = None
) -> None:
    """Evaluate model performance on code translation task."""
    # Create temporary sandbox directory
    workspace_dir = Path(tempfile.mkdtemp())
    results_dir = Path("results") / f"{model}_{source_lang}_to_{target_lang}"
    os.makedirs(results_dir, exist_ok=True)

    print("Set workspace dir to", workspace_dir)
    print("Set results dir to", results_dir)

    # Initialize sample collector with concrete implementation
    collector = ExercismSampleCollector(
        source_lang=source_lang, target_lang=target_lang, workspace_dir=workspace_dir
    )

    file_manager = FileManager(workspace_dir / "python", workspace_dir / "go")
    sandbox = Sandbox(workspace_dir, target_lang)

    # Create tasks for parallel processing
    tasks = []
    for sample in collector.get_code_samples(num_samples):
        task = process_sample(
            sample,
            workspace_dir,
            file_manager,
            sandbox,
            model,
            source_lang,
            target_lang,
            results_dir,
        )
        tasks.append(task)

    # Run all tasks concurrently
    await asyncio.gather(*tasks)


@click.group()
def cli():
    """Benchmark different LLMs for code translation tasks."""
    pass


@cli.command()
@click.option(
    "--models",
    "-m",
    multiple=True,
    type=click.Choice(SUPPORTED_MODELS, case_sensitive=False),
    required=True,
    help="List of LLM models to benchmark",
)
@click.option(
    "--source",
    "-s",
    type=click.Choice(SUPPORTED_LANGUAGES, case_sensitive=False),
    default="python",
    help="Source programming language (default: python)",
)
@click.option(
    "--target",
    "-t",
    type=click.Choice(SUPPORTED_LANGUAGES, case_sensitive=False),
    default="go",
    help="Target programming language (default: go)",
)
@click.option(
    "--num-samples",
    "-n",
    type=int,
    default=None,
    help="Number of samples to process (default: all available)",
)
def translate(models, source, target, num_samples):
    """Translate code from one programming language to another using specified LLMs."""
    click.echo(f"Models selected: {', '.join(models)}")
    click.echo(f"Source language: {source}")
    click.echo(f"Target language: {target}")
    click.echo(f"Number of samples: {'all' if num_samples is None else num_samples}")

    for model in models:
        click.echo(f"\nTranslating with {model}...")
        asyncio.run(evaluate(model, source, target, num_samples))


if __name__ == "__main__":
    cli()
