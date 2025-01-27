import abc
import asyncio
import glob
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Tuple

import click

from particle.file_manager import FileManager
from particle.llm_translator import LLMTranslator

SUPPORTED_MODELS = [
    "gpt-4",
    "gpt-3.5-turbo",
    "claude-3-sonnet",
    "gemini-1.5-pro",
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
    test_code: str
    test_path: str


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

            # Source code path (example implementation)
            source_file = source_ex_path / ".meta" / f"example.{source_ext}"
            if not source_file.exists():
                continue

            # Test code path
            test_file = target_ex_path / f"{exercise_name}_test.{target_ext}"
            if not test_file.exists():
                continue

            try:
                # Read source and test files
                with open(source_file) as f:
                    source_code = f.read()
                with open(test_file) as f:
                    test_code = f.read()

                # Create relative paths from workspace directory
                source_rel_path = source_file.relative_to(self.workspace_dir)
                test_rel_path = test_file.relative_to(self.workspace_dir)

                yield CodeSample(
                    source_code=source_code,
                    source_path=str(source_rel_path),
                    test_code=test_code,
                    test_path=str(test_rel_path),
                )
                samples_yielded += 1

            except (IOError, OSError) as e:
                print(f"Error processing {exercise_name}: {e}")
                continue


class TestRunner:
    """Handles language-specific test running logic."""

    @staticmethod
    def get_test_command(lang: str, test_file: Path) -> Tuple[list[str], dict]:
        """Returns the command to run tests and any necessary environment setup."""
        test_dir = test_file.parent

        commands = {
            "python": (
                ["python", "-m", "pytest", str(test_file)],
                {"env": {**os.environ, "PYTHONPATH": str(test_dir)}},
            ),
            "go": (
                ["go", "test", str(test_dir)],
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

        return commands.get(lang, ([], {}))

    @staticmethod
    def setup_test_environment(lang: str, test_dir: Path) -> None:
        """Set up any necessary test environment for the given language."""
        if lang == "python":
            # Create requirements.txt if needed
            requirements = test_dir / "requirements.txt"
            if not requirements.exists():
                with open(requirements, "w") as f:
                    f.write("pytest\n")
            subprocess.run(["pip", "install", "-r", str(requirements)], check=True)

        elif lang == "go":
            # Initialize go module if needed
            if not (test_dir / "go.mod").exists():
                module_name = f"exercism/{test_dir.name}"
                subprocess.run(
                    ["go", "mod", "init", module_name], cwd=test_dir, check=True
                )
                # Download test dependencies
                subprocess.run(["go", "mod", "tidy"], cwd=test_dir, check=True)

        elif lang in ["typescript", "rust", "java", "cpp"]:
            raise NotImplementedError(
                f"Test environment setup for {lang} is not yet implemented"
            )

        else:
            raise ValueError(f"Unsupported language: {lang}")

    @staticmethod
    def run_tests(lang: str, test_file: Path) -> subprocess.CompletedProcess:
        """Run tests for the given language and return the result."""
        # Set up test environment
        test_dir = test_file.parent
        TestRunner.setup_test_environment(lang, test_dir)

        # Get test command for target language
        cmd, run_kwargs = TestRunner.get_test_command(lang, test_file)
        if not cmd:
            raise ValueError(f"Unsupported target language: {lang}")

        # Run tests
        result = subprocess.run(cmd, capture_output=True, text=True, **run_kwargs)

        return result


async def process_sample(
    sample: CodeSample,
    sandbox_dir: Path,
    model: str,
    source_lang: str,
    target_lang: str,
    results_dir: Path,
) -> None:
    """Process a single code sample."""
    # Create necessary directories
    source_file = sandbox_dir / sample.source_path
    test_file = sandbox_dir / sample.test_path
    translated_file = (
        test_file.parent
        / f"{test_file.stem.replace('_test', '')}.{ExercismSampleCollector.LANGUAGE_EXTENSIONS[target_lang]}"
    )

    os.makedirs(os.path.dirname(source_file), exist_ok=True)
    os.makedirs(os.path.dirname(test_file), exist_ok=True)

    # Write source and test files
    with open(source_file, "w") as f:
        f.write(sample.source_code)
    with open(test_file, "w") as f:
        f.write(sample.test_code)

    try:
        # Initialize translator and translate code
        file_manager = FileManager(sandbox_dir)
        translator = LLMTranslator(model, file_manager)

        # Create a dictionary mapping source file to code snippets
        code_snippets = {str(source_file): [sample.source_code]}

        # Translate the code
        translated_code = translator.translate(code_snippets, source_file.parent.name)

        # Write translated code
        with open(translated_file, "w") as f:
            f.write(translated_code)

        # Run tests using TestRunner
        result = TestRunner.run_tests(target_lang, test_file)

        # Create results file
        exercise_name = source_file.parent.name
        result_file = results_dir / f"{exercise_name}_results.txt"

        with open(result_file, "w") as f:
            f.write(f"Exercise: {exercise_name}\n")
            f.write(f"Source Language: {source_lang}\n")
            f.write(f"Target Language: {target_lang}\n")
            f.write(f"Model: {model}\n")
            f.write(f"Test Return Code: {result.returncode}\n\n")

            if result.returncode != 0:
                f.write("Test Output (stderr):\n")
                f.write(result.stderr)
                f.write("\nTest Output (stdout):\n")
                f.write(result.stdout)

                # If tests failed, try to translate again with the error output
                translated_code = translator.retry(
                    result.stderr + "\n" + result.stdout, source_file.parent.name
                )

                # Write the retried translation
                with open(translated_file, "w") as f_retry:
                    f_retry.write(translated_code)

                # Run tests again
                retry_result = TestRunner.run_tests(target_lang, test_file)

                f.write("\n\nRetry attempt:\n")
                f.write(f"Test Return Code: {retry_result.returncode}\n")
                if retry_result.returncode != 0:
                    f.write("Test Output (stderr):\n")
                    f.write(retry_result.stderr)
                    f.write("\nTest Output (stdout):\n")
                    f.write(retry_result.stdout)
                else:
                    f.write("Tests passed successfully after retry!\n")
            else:
                f.write("Tests passed successfully!\n")

    except subprocess.CalledProcessError as e:
        # Handle setup failures
        error_file = results_dir / f"{exercise_name}_error.txt"
        with open(error_file, "w") as f:
            f.write(f"Error setting up/running tests: {str(e)}\n")
            if e.stdout:
                f.write(f"\nStdout:\n{e.stdout.decode()}")
            if e.stderr:
                f.write(f"\nStderr:\n{e.stderr.decode()}")

    except Exception as e:
        # Handle other errors
        error_file = results_dir / f"{exercise_name}_error.txt"
        with open(error_file, "w") as f:
            f.write(f"Unexpected error: {str(e)}\n")


async def evaluate(
    model: str, source_lang: str, target_lang: str, num_samples: int = None
) -> None:
    """Evaluate model performance on code translation task."""
    # Create temporary sandbox directory
    with tempfile.TemporaryDirectory() as temp_dir:
        sandbox_dir = Path(temp_dir)
        results_dir = Path("results") / f"{model}_{source_lang}_to_{target_lang}"
        os.makedirs(results_dir, exist_ok=True)

        # Initialize sample collector with concrete implementation
        collector = ExercismSampleCollector(
            source_lang=source_lang, target_lang=target_lang, workspace_dir=sandbox_dir
        )

        # Create tasks for parallel processing
        tasks = []
        for sample in collector.get_code_samples(num_samples):
            task = process_sample(
                sample,
                sandbox_dir,
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
