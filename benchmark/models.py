from django.db import models


class BenchmarkRun(models.Model):
    """Model representing a benchmark run with metadata."""

    name = models.CharField(
        max_length=100,
        blank=True,
        help_text="Random word identifier for this benchmark run",
    )
    model_name = models.CharField(max_length=100)
    source_lang = models.CharField(max_length=50)
    target_lang = models.CharField(max_length=50)
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True, blank=True)

    def __str__(self) -> str:
        return f"{self.name} - {self.source_lang} to {self.target_lang} - {self.model_name} ({self.start_time.strftime('%Y-%m-%d %H:%M')})"


class ExerciseResult(models.Model):
    """Model representing results for a single exercise in a benchmark run."""

    benchmark_run = models.ForeignKey(
        BenchmarkRun, on_delete=models.CASCADE, related_name="exercise_results"
    )
    exercise_name = models.CharField(max_length=200)
    num_retries = models.IntegerField(default=0)
    return_code = models.IntegerField()
    output_content = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ["benchmark_run", "exercise_name"]

    def __str__(self) -> str:
        status = "SUCCESS" if self.return_code == 0 else "FAILED"
        return f"{self.benchmark_run.name} - {self.exercise_name} - {status} - {self.num_retries} retries"
