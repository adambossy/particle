import abc
import asyncio
import os
import subprocess
import tempfile
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Generator

import aiofiles
import click

from particle.file_manager import FileManager
from particle.llm_translator import LLMTranslator

SUPPORTED_MODELS = [
    "gpt-4o-2024-08-06",
    "o3-mini",
    "anthropic/claude-3-5-sonnet-20241022",
    "anthropic/claude-3-7-sonnet-20250219",
    "deepseek/deepseek-chat",
    "deepseek/deepseek-coder",
    "fireworks_ai/accounts/fireworks/models/deepseek-v3",
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
    extra_test_code: list[str] = field(default_factory=list)
    extra_test_paths: list[str] = field(default_factory=list)


class SampleCollector(abc.ABC):
    @abc.abstractmethod
    async def get_code_samples(self) -> AsyncGenerator[CodeSample, None]:
        """Returns an async generator of CodeSample instances."""
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
        exercises_glob = list(source_exercises.glob("*"))
        print(f"Found {len(exercises_glob)} exercises in source repo")
        for source_ex_path in source_exercises.glob("*"):
            if not source_ex_path.is_dir():
                continue

            exercise_name = source_ex_path.name
            target_ex_path = target_exercises / exercise_name

            # Check if the same exercise exists in target repo
            if target_ex_path.exists():
                yield source_ex_path, target_ex_path

    async def get_code_samples(
        self, num_samples: int | None = None
    ) -> AsyncGenerator[CodeSample, None]:
        """Generate CodeSample instances for matching exercises."""
        source_ext = self.LANGUAGE_EXTENSIONS[self.source_lang]
        target_ext = self.LANGUAGE_EXTENSIONS[self.target_lang]

        samples_yielded = 0

        for source_ex_path, target_ex_path in self._get_exercise_pairs():
            if num_samples is not None and samples_yielded >= num_samples:
                break

            exercise_name = source_ex_path.name
            exercise_name_snake = exercise_name.replace("-", "_")

            # Contains solution code to translate
            source_file = source_ex_path / ".meta" / f"example.{source_ext}"
            if not source_file.exists():
                print(
                    f"Skipping {exercise_name} because source file {source_file} doesn't exist in source repo"
                )
                continue

            # Contains interface stub
            source_interface = source_ex_path / f"{exercise_name_snake}.{source_ext}"
            if not source_interface.exists():
                print(
                    f"Skipping {exercise_name} because interface file {source_interface} doesn't exist in source repo"
                )
                continue

            # Contains tests to help translation along
            source_test_file = (
                source_ex_path / f"{exercise_name_snake}_test.{source_ext}"
            )
            if not source_test_file.exists():
                print(
                    f"Skipping {exercise_name} because test file {source_test_file} doesn't exist in source repo"
                )
                continue

            # Contains interface stub
            target_code_file = target_ex_path / f"{exercise_name_snake}.{target_ext}"
            if not target_code_file.exists():
                print(
                    f"Skipping {exercise_name} because target code file {target_code_file} doesn't exist in target repo"
                )
                continue

            # Contains tests that translated code must pass
            target_test_file = (
                target_ex_path / f"{exercise_name_snake}_test.{target_ext}"
            )
            if not target_test_file.exists():
                print(
                    f"Skipping {exercise_name} because target test file {target_test_file} doesn't exist in target repo"
                )
                continue

            # Collect additional test files with a different prefix
            extra_test_files = []
            for test_file in target_ex_path.glob(f"*_test.{target_ext}"):
                prefix = test_file.stem.split("_test")[0]
                if prefix != exercise_name_snake:
                    extra_test_files.append(
                        test_file.relative_to(self.target_repo_path)
                    )

            # Store the relative paths of these extra test files
            extra_test_paths = [str(path) for path in extra_test_files]

            print(
                f"Found {len(extra_test_paths)} extra test files for {exercise_name}: {extra_test_paths}"
            )

            try:
                # Read source and test files asynchronously
                async with aiofiles.open(source_file) as f:
                    source_code = await f.read()
                async with aiofiles.open(source_interface) as f:
                    source_interface_code = await f.read()
                async with aiofiles.open(source_test_file) as f:
                    source_test_code = await f.read()
                async with aiofiles.open(target_code_file) as f:
                    target_interface = await f.read()
                async with aiofiles.open(target_test_file) as f:
                    target_test_code = await f.read()

                extra_test_code = []
                for extra_test_path in extra_test_paths:
                    full_path = self.target_repo_path / extra_test_path
                    async with aiofiles.open(full_path) as f:
                        extra_test_code.append(await f.read())

                # Create relative paths from workspace directory
                source_rel_path = source_file.relative_to(self.source_repo_path)
                source_test_rel_path = source_test_file.relative_to(
                    self.source_repo_path
                )
                target_rel_path = target_code_file.relative_to(self.target_repo_path)
                test_rel_path = target_test_file.relative_to(self.target_repo_path)

                yield CodeSample(
                    source_code=source_code,
                    source_path=str(source_rel_path),
                    source_interface=source_interface_code,
                    source_test_code=source_test_code,
                    source_test_path=str(source_test_rel_path),
                    target_interface=target_interface,
                    target_test_code=target_test_code,
                    target_path=str(target_rel_path),
                    target_test_path=str(test_rel_path),
                    extra_test_code=extra_test_code,
                    extra_test_paths=extra_test_paths,
                )
                samples_yielded += 1

            except (IOError, OSError) as e:
                print(f"Error processing {exercise_name}: {e}")
                print(f"Current working directory: {os.getcwd()}")
                raise e


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

    async def create_code_file(self, rel_path: Path | str, source_code: str) -> Path:
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
        async with aiofiles.open(file_path, "w") as f:
            await f.write(source_code)
        return file_path

    async def create_test_file(self, rel_path: Path | str, test_code: str) -> Path:
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
        async with aiofiles.open(file_path, "w") as f:
            await f.write(test_code)
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

    async def setup_test_environment(self, test_dir: Path) -> None:
        """Set up any necessary test environment for the given language."""
        if self.target_lang == "python":
            # Create requirements.txt if needed
            requirements = test_dir / "requirements.txt"
            if not requirements.exists():
                async with aiofiles.open(requirements, "w") as f:
                    await f.write("pytest\n")
            subprocess.run(["pip", "install", "-r", str(requirements)], check=True)

        elif self.target_lang == "go":
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

    async def run_tests(self, test_file: Path) -> subprocess.CompletedProcess:
        """Run tests for the given language and return the result."""
        # Set up test environment
        test_dir = test_file.parent
        await self.setup_test_environment(test_dir)

        # Get test command for target language
        cmd, run_kwargs = self.get_test_command(test_file)
        if not cmd:
            raise ValueError(f"Unsupported target language: {self.target_lang}")

        # Run the command in an executor to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: subprocess.run(
                cmd, capture_output=True, text=True, cwd=run_kwargs["cwd"]
            ),
        )

        result.stdout = result.stdout or ""
        result.stderr = result.stderr or ""

        return result


async def process_sample(
    sample: CodeSample,
    workspace_dir: Path,
    file_manager: FileManager,
    sandbox: Sandbox,
    model: str,
    results_dir: Path,
    output_dir: Path,
) -> tuple[str, int, int]:
    """Process a single code sample."""
    # Create necessary directories
    # TODO (adam) Have Sandbox manage the portion before the try/except block
    test_file = workspace_dir / "sandbox" / sample.target_test_path
    extra_test_files = []
    for extra_test_path in sample.extra_test_paths:
        extra_test_files.append(workspace_dir / "sandbox" / extra_test_path)
    translated_file = workspace_dir / "sandbox" / sample.target_path

    # TODO (adam) move into CodeSample object
    exercise_name = translated_file.parent.name
    result_file = results_dir / f"{exercise_name}.txt"
    output_file = output_dir / f"{exercise_name}.txt"

    print(f"Processing sample {exercise_name}")

    os.makedirs(os.path.dirname(test_file), exist_ok=True)

    for extra_test_file in extra_test_files:
        os.makedirs(os.path.dirname(extra_test_file), exist_ok=True)

    # Write source and test files
    async with aiofiles.open(test_file, "w") as f:
        await f.write(sample.target_test_code)
    for extra_test_file, extra_test_code in zip(
        extra_test_files, sample.extra_test_code
    ):
        async with aiofiles.open(extra_test_file, "w") as f:
            await f.write(extra_test_code)

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

    num_retries = 0
    result = None
    output_content = []

    try:
        # Record initial state
        output_content.append("=== Initial Source Code ===")
        output_content.append(f"Source file: {sample.source_path}")
        output_content.append(sample.source_code)
        output_content.append("\n=== Source Interface ===")
        output_content.append(sample.source_interface)
        output_content.append("\n=== Target Interface ===")
        output_content.append(sample.target_interface)
        output_content.append("\n=== Source Test Code ===")
        output_content.append(sample.source_test_code)
        output_content.append("\n=== Target Test Code ===")
        output_content.append(sample.target_test_code)

        # Translate the code
        print(f"Fetching translation for {exercise_name}")
        translated_code = await translator.translate(
            code_snippets,
            special_instructions=special_instructions,
        )

        # Record translation
        output_content.append("\n=== Initial Translation ===")
        output_content.append(translated_code)

        # Write translated code
        async with aiofiles.open(translated_file, "w") as f:
            await f.write(translated_code)

        # Run tests using TestRunner
        print(f"Running tests for {exercise_name}")
        result = await sandbox.run_tests(test_file)
        output_content.append("\n=== Initial Test Results ===")
        output_content.append(f"Return code: {result.returncode}")
        output_content.append("=== STDOUT ===")
        output_content.append(result.stdout)
        output_content.append("=== STDERR ===")
        output_content.append(result.stderr)

        while result.returncode != 0 and num_retries < 10:
            # If tests failed, try to translate again with the error output
            last_test_output = result.stderr + "\n" + result.stdout
            print(f"Retrying translation for {exercise_name}")
            translated_code = await translator.retry(
                last_test_output,
                sample.target_test_code,
            )

            # Record retry attempt
            output_content.append(f"\n=== Retry Attempt {num_retries + 1} ===")
            output_content.append(translated_code)

            # Write the retried translation
            async with aiofiles.open(translated_file, "w") as f:
                await f.write(translated_code)

            # Run tests again
            print(f"Running tests again for {exercise_name}")
            result = await sandbox.run_tests(test_file)
            output_content.append(f"\n=== Test Results (Attempt {num_retries + 1}) ===")
            output_content.append(f"Return code: {result.returncode}")
            output_content.append("=== STDOUT ===")
            output_content.append(result.stdout)
            output_content.append("=== STDERR ===")
            output_content.append(result.stderr)

            num_retries += 1

        if result.returncode == 0:
            print(f"Tests passed successfully after {num_retries} retries!")
            output_content.append("\n=== FINAL STATUS: SUCCESS ===")
        else:
            print(f"Tests failed after {num_retries} retries!")
            output_content.append("\n=== FINAL STATUS: FAILED ===")

        # Write detailed output
        async with aiofiles.open(output_file, "w") as f:
            print(f"Writing output to {output_file}")
            await f.write("\n".join(output_content))

        # Write summary results
        async with aiofiles.open(result_file, "w") as f:
            print(f"Writing result to {result_file}")
            await f.write(f"{exercise_name},{num_retries},{result.returncode}")

        print(f"Finished processing sample {exercise_name}")

        return exercise_name, num_retries, result.returncode

    except Exception as e:
        traceback.print_exc()

        error_msg = (
            f"Error processing sample (returncode={result and result.returncode}): {e}"
        )
        print(error_msg)

        # Record error in output
        output_content.append("\n=== ERROR ===")
        output_content.append(error_msg)
        output_content.append("\n=== FINAL STATUS: ERROR ===")

        # Write detailed output even in case of error
        async with aiofiles.open(output_file, "w") as f:
            await f.write("\n".join(output_content))

        # Write summary results
        async with aiofiles.open(result_file, "w") as f:
            await f.write(f"{exercise_name},{num_retries},1")

        return exercise_name, num_retries, 1


async def collect_results(results_dir: Path) -> None:
    """Collect and aggregate results from individual exercise files into a single results.txt file.

    Args:
        results_dir: Path to the directory containing individual result files
    """
    results_file = results_dir / "results.txt"

    # Write CSV header
    async with aiofiles.open(results_file, "w") as f:
        await f.write("Exercise Name,Num Attempts,Return Code\n")

        # Read and aggregate all result files except results.txt
        for file_path in results_dir.glob("*.txt"):
            if file_path.name != "results.txt":
                async with aiofiles.open(file_path) as f_result:
                    content = await f_result.read()
                    await f.write(f"{content.strip()}\n")


async def limited_gather(tasks: list[asyncio.Task], limit: int) -> list:
    semaphore = asyncio.Semaphore(limit)

    async def sem_task(task):
        async with semaphore:
            return await task

    return await asyncio.gather(*(sem_task(task) for task in tasks))


async def evaluate(
    model: str,
    source_lang: str,
    target_lang: str,
    num_samples: int = None,
    parallel: bool = True,
) -> None:
    """Evaluate model performance on code translation task."""
    # Create temporary sandbox directory
    workspace_dir = Path(tempfile.mkdtemp())

    # Get current date and time
    current_time = datetime.now().strftime("%Y%m%d_%I%M%p")
    model_name = model.split("/")[-1]

    # Create base directory structure
    base_dir = Path(f"results/{source_lang}_to_{target_lang}_{current_time}")
    model_dir = base_dir / model_name
    results_dir = model_dir / "results"
    output_dir = model_dir / "output"

    # Create all directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("Set workspace dir to", workspace_dir)
    print("Set results dir to", results_dir)
    print("Set output dir to", output_dir)

    # Initialize sample collector with concrete implementation
    collector = ExercismSampleCollector(
        source_lang=source_lang,
        target_lang=target_lang,
        workspace_dir=workspace_dir,
    )

    file_manager = FileManager(workspace_dir / "python", workspace_dir / "go")
    sandbox = Sandbox(workspace_dir, target_lang)

    # Collect all samples first
    samples = []
    async for sample in collector.get_code_samples(num_samples):
        samples.append(sample)

    print(f"Collected {len(samples)} samples for processing")

    results = []

    if parallel:
        # Create tasks for parallel processing
        tasks = []
        for sample in samples:
            print(f"Creating task for sample: {sample.source_path}")
            task = process_sample(
                sample,
                workspace_dir,
                file_manager,
                sandbox,
                model,
                results_dir,
                output_dir,
            )
            tasks.append(task)

        # Use the limited_gather function instead of asyncio.gather
        results = await limited_gather(tasks, limit=8)  # Adjust 'limit' as needed
        print(f"Processed {len(results)} samples in parallel")
    else:
        # Process samples serially
        for sample in samples:
            print(f"Processing sample: {sample.source_path}")
            result = await process_sample(
                sample,
                workspace_dir,
                file_manager,
                sandbox,
                model,
                results_dir,
                output_dir,
            )
            results.append(result)
        print(f"Processed {len(results)} samples serially")

    # Collect and aggregate results
    await collect_results(results_dir)


def get_latest_results_dir() -> Path:
    """Get the most recently created results directory.

    Returns:
        Path to the most recent results directory
    """
    results_dir = Path("results")
    if not results_dir.exists():
        raise click.ClickException("No results directory found")

    dirs = [(d.stat().st_mtime, d) for d in results_dir.iterdir() if d.is_dir()]
    if not dirs:
        raise click.ClickException("No result directories found")

    return sorted(dirs, key=lambda x: x[0], reverse=True)[0][1]


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
@click.option(
    "--parallel",
    "-p",
    is_flag=True,
    default=False,
    help="Process samples in parallel (default: serial)",
)
def translate(models, source, target, num_samples, parallel):
    """Translate code from one programming language to another using specified LLMs."""
    click.echo(f"Models selected: {', '.join(models)}")
    click.echo(f"Source language: {source}")
    click.echo(f"Target language: {target}")
    click.echo(f"Number of samples: {'all' if num_samples is None else num_samples}")
    click.echo(f"Processing mode: {'parallel' if parallel else 'serial'}")

    for model in models:
        click.echo(f"\nTranslating with {model}...")
        asyncio.run(evaluate(model, source, target, num_samples, parallel))


@cli.command()
@click.option(
    "--directory",
    "-d",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory containing results to collect (default: most recent)",
)
def collect(directory: Path | None) -> None:
    """Collect and aggregate results from individual exercise files.

    If no directory is specified, uses the most recently created results directory.
    """
    if directory is None:
        directory = get_latest_results_dir()

    click.echo(f"Collecting results from {directory}")
    asyncio.run(collect_results(directory))
    click.echo("Results collected to results.txt")


if __name__ == "__main__":
    cli()
