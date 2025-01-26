import abc
import asyncio
import glob
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List

import click

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

    def get_code_samples(self) -> Generator[CodeSample, None, None]:
        """Generate CodeSample instances for matching exercises."""
        source_ext = self.LANGUAGE_EXTENSIONS[self.source_lang]
        target_ext = self.LANGUAGE_EXTENSIONS[self.target_lang]

        for source_ex_path, target_ex_path in self._get_exercise_pairs():
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

            except (IOError, OSError) as e:
                print(f"Error processing {exercise_name}: {e}")
                continue


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
    os.makedirs(os.path.dirname(source_file), exist_ok=True)
    os.makedirs(os.path.dirname(test_file), exist_ok=True)

    # Write source and test files
    with open(source_file, "w") as f:
        f.write(sample.source_code)
    with open(test_file, "w") as f:
        f.write(sample.test_code)

    # TODO: Implement test running logic
    # TODO: Implement translation logic
    # TODO: Write results to results_dir


async def evaluate(model: str, source_lang: str, target_lang: str) -> None:
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
        for sample in collector.get_code_samples():
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
def translate(models, source, target):
    """Translate code from one programming language to another using specified LLMs."""
    click.echo(f"Models selected: {', '.join(models)}")
    click.echo(f"Source language: {source}")
    click.echo(f"Target language: {target}")

    for model in models:
        click.echo(f"\nTranslating with {model}...")
        asyncio.run(evaluate(model, source, target))


if __name__ == "__main__":
    cli()
