from typing import List, Optional, Type

from django.contrib import admin

from .models import BenchmarkRun, ExerciseResult


class ExerciseResultInline(admin.TabularInline):
    """Inline admin for ExerciseResult model."""

    model: Type[ExerciseResult] = ExerciseResult
    extra: int = 0
    readonly_fields: List[str] = ["created_at", "benchmark_name"]
    fields: List[str] = [
        "exercise_name",
        "benchmark_name",
        "return_code",
        "num_retries",
        "created_at",
    ]
    can_delete: bool = False


@admin.register(BenchmarkRun)
class BenchmarkRunAdmin(admin.ModelAdmin):
    """Admin configuration for BenchmarkRun model."""

    list_display: List[str] = [
        "name",
        "get_model_names",
        "source_lang",
        "target_lang",
        "start_time",
        "end_time",
    ]
    list_filter: List[str] = ["name", "source_lang", "target_lang"]
    search_fields: List[str] = ["name", "source_lang", "target_lang"]
    readonly_fields: List[str] = ["name", "model_names", "start_time"]
    date_hierarchy: str = "start_time"
    inlines = [ExerciseResultInline]

    def get_model_names(self, obj: BenchmarkRun) -> str:
        """Get the model names as a comma-separated string."""
        return ", ".join(obj.model_names) if obj.model_names else "No models"

    get_model_names.short_description = "Models"


@admin.register(ExerciseResult)
class ExerciseResultAdmin(admin.ModelAdmin):
    """Admin configuration for ExerciseResult model."""

    list_display: List[str] = [
        "exercise_name",
        "benchmark_name",
        "benchmark_run",
        "return_code",
        "num_retries",
        "created_at",
    ]
    list_filter: List[str] = [
        "return_code",
        "exercise_name",
        "benchmark_run__name",
    ]
    search_fields: List[str] = [
        "exercise_name",
        "benchmark_run__name",
    ]
    readonly_fields: List[str] = ["created_at", "benchmark_name"]
    date_hierarchy: str = "created_at"
    raw_id_fields: List[str] = ["benchmark_run"]
