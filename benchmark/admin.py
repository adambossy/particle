from typing import List, Optional, Type

from django.contrib import admin

from .models import BenchmarkRun, ExerciseResult


@admin.register(BenchmarkRun)
class BenchmarkRunAdmin(admin.ModelAdmin):
    """Admin configuration for BenchmarkRun model."""

    list_display: List[str] = [
        "name",
        "model_name",
        "source_lang",
        "target_lang",
        "start_time",
        "end_time",
    ]
    list_filter: List[str] = ["name", "model_name", "source_lang", "target_lang"]
    search_fields: List[str] = ["name", "model_name", "source_lang", "target_lang"]
    readonly_fields: List[str] = ["name", "start_time"]
    date_hierarchy: str = "start_time"


class ExerciseResultInline(admin.TabularInline):
    """Inline admin for ExerciseResult model."""

    model: Type[ExerciseResult] = ExerciseResult
    extra: int = 0
    readonly_fields: List[str] = ["created_at"]
    fields: List[str] = ["exercise_name", "return_code", "num_retries", "created_at"]
    can_delete: bool = False


@admin.register(ExerciseResult)
class ExerciseResultAdmin(admin.ModelAdmin):
    """Admin configuration for ExerciseResult model."""

    list_display: List[str] = [
        "exercise_name",
        "get_benchmark_name",
        "benchmark_run",
        "return_code",
        "num_retries",
        "created_at",
    ]
    list_filter: List[str] = [
        "return_code",
        "exercise_name",
        "benchmark_run__name",
        "benchmark_run__model_name",
    ]
    search_fields: List[str] = [
        "exercise_name",
        "benchmark_run__name",
        "benchmark_run__model_name",
    ]
    readonly_fields: List[str] = ["created_at"]
    date_hierarchy: str = "created_at"
    raw_id_fields: List[str] = ["benchmark_run"]

    def get_benchmark_name(self, obj: ExerciseResult) -> str:
        """Get the name of the benchmark run."""
        return obj.benchmark_run.name

    get_benchmark_name.short_description = "Benchmark Name"
    get_benchmark_name.admin_order_field = "benchmark_run__name"
