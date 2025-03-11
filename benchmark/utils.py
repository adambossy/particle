import os
import random
import subprocess
from typing import Any, Dict, Optional

from asgiref.sync import sync_to_async
from django.utils import timezone

from benchmark.models import BenchmarkRun, ExerciseResult

# Fallback to a small list of words if no dictionary is found
fallback_words = [
    "apple",
    "banana",
    "cherry",
    "date",
    "elderberry",
    "fig",
    "grape",
    "honeydew",
    "kiwi",
    "lemon",
    "mango",
    "nectarine",
    "orange",
    "papaya",
    "quince",
    "raspberry",
    "strawberry",
    "tangerine",
    "watermelon",
    "cosmic",
    "stellar",
    "lunar",
    "solar",
    "galactic",
    "quantum",
    "atomic",
    "nebula",
    "comet",
    "asteroid",
]


def get_random_word() -> str:
    """
    Get a random word from the system dictionary.

    Returns:
        str: A random word from the system dictionary
    """
    # Default dictionary paths for common Unix systems
    dictionary_paths = [
        "/usr/share/dict/words",
        "/usr/dict/words",
        "/etc/dictionaries-common/words",
    ]

    # Find the first available dictionary
    dictionary_path = None
    for path in dictionary_paths:
        if os.path.exists(path):
            dictionary_path = path
            break

    if not dictionary_path:
        return random.choice(fallback_words)

    # Get a random word from the dictionary
    try:
        # Use grep to filter out words with apostrophes and other special characters
        result = subprocess.run(
            f'grep -v "[^a-zA-Z]" {dictionary_path} | shuf -n 1',
            shell=True,
            capture_output=True,
            text=True,
            check=True,
        )
        word = result.stdout.strip()

        # If the word is too long, try again with a length filter
        if len(word) > 10:
            result = subprocess.run(
                f'grep -v "[^a-zA-Z]" {dictionary_path} | grep -E "^.{{3,10}}$" | shuf -n 1',
                shell=True,
                capture_output=True,
                text=True,
                check=True,
            )
            word = result.stdout.strip()

        # Capitalize the word
        return word.capitalize() if word else "Benchmark"
    except subprocess.CalledProcessError:
        # Fallback to a random word from our list
        return random.choice(fallback_words)


@sync_to_async
def create_benchmark_run(
    models: list[str], source_lang: str, target_lang: str
) -> Dict[str, Any]:
    """
    Create a new benchmark run record using Django ORM.

    Args:
        models: List of LLM model names being benchmarked
        source_lang: The source programming language
        target_lang: The target programming language

    Returns:
        Dict[str, Any]: The created record data as a dictionary
    """
    # Generate a random word for the benchmark name
    random_word = get_random_word()

    benchmark_run = BenchmarkRun.objects.create(
        name=random_word,
        model_names=models,
        source_lang=source_lang,
        target_lang=target_lang,
    )

    return {
        "id": benchmark_run.id,
        "name": benchmark_run.name,
        "model_names": benchmark_run.model_names,
        "source_lang": benchmark_run.source_lang,
        "target_lang": benchmark_run.target_lang,
        "start_time": benchmark_run.start_time.isoformat(),
    }


@sync_to_async
def create_exercise_result(
    benchmark_run_id: int,
    exercise_name: str,
    num_retries: int,
    return_code: int,
    output_content: Optional[str] = None,
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create an exercise result record using Django ORM.

    Args:
        benchmark_run_id: The ID of the parent benchmark run
        exercise_name: The name of the exercise
        num_retries: Number of retry attempts
        return_code: The return code (0=success, non-zero=failure)
        output_content: Optional output content from the test run
        error_message: Optional error message if an exception occurred

    Returns:
        Dict[str, Any]: The created record data as a dictionary
    """
    benchmark_run = BenchmarkRun.objects.get(id=benchmark_run_id)
    exercise_result = ExerciseResult.objects.create(
        benchmark_run=benchmark_run,
        exercise_name=exercise_name,
        num_retries=num_retries,
        return_code=return_code,
        output_content=output_content or error_message,
    )

    return {
        "id": exercise_result.id,
        "benchmark_run_id": benchmark_run_id,
        "benchmark_name": benchmark_run.name,
        "exercise_name": exercise_result.exercise_name,
        "num_retries": exercise_result.num_retries,
        "return_code": exercise_result.return_code,
        "created_at": exercise_result.created_at.isoformat(),
    }


@sync_to_async
def update_benchmark_run_end_time(benchmark_run_id: int) -> None:
    """
    Update the end_time of a benchmark run.

    Args:
        benchmark_run_id: The ID of the benchmark run to update
    """
    benchmark_run = BenchmarkRun.objects.get(id=benchmark_run_id)
    benchmark_run.end_time = timezone.now()
    benchmark_run.save()
