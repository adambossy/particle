from typing import Any, Dict, Optional

from asgiref.sync import sync_to_async
from django.utils import timezone

from benchmark.models import BenchmarkRun, ExerciseResult


@sync_to_async
def create_benchmark_run(
    model_name: str, source_lang: str, target_lang: str
) -> Dict[str, Any]:
    """
    Create a new benchmark run record using Django ORM.

    Args:
        model_name: The name of the LLM model being benchmarked
        source_lang: The source programming language
        target_lang: The target programming language

    Returns:
        Dict[str, Any]: The created record data as a dictionary
    """
    benchmark_run = BenchmarkRun.objects.create(
        model_name=model_name,
        source_lang=source_lang,
        target_lang=target_lang,
    )

    return {
        "id": benchmark_run.id,
        "model_name": benchmark_run.model_name,
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
