import abc
import asyncio
import os
import shutil
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

        # Initialize sample collector (implementation to be provided)
        collector = (
            SampleCollector()
        )  # This line needs to be updated with concrete implementation

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
