from collections import Counter
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def visualize(results: Dict[str, List[Tuple[str, int, int]]]) -> None:
    # Process data
    model_stats = {}
    for model, exercises in results.items():
        total_count = len(exercises)
        # Filter successful exercises (status code 0)
        successful = [ex for ex in exercises if ex[2] == 0]
        # Group by retry count and calculate percentages
        retry_counts = Counter(ex[1] for ex in successful)

        # Sort by retry count and calculate percentages
        sorted_retries = sorted(retry_counts.items())
        percentages = [(count / total_count * 100) for _, count in sorted_retries]

        model_stats[model] = {
            "retries": [r for r, _ in sorted_retries],
            "percentages": percentages,
            "counts": [c for _, c in sorted_retries],
            "total": total_count,
        }

        # Print statistics
        print(f"\nModel: {model}")
        for retries, count in sorted_retries:
            print(f"Retry group {retries}: {count} exercises")
        print(f"Total exercises: {total_count}")

    # Plotting
    models = list(results.keys())
    x = np.arange(len(models))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create stacked bars
    bottom = np.zeros(len(models))
    # Replace viridis colormap with custom colors
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
    ]  # Blue to orange gradient from image

    for retry_idx in range(
        max(len(stats["retries"]) for stats in model_stats.values())
    ):
        heights = []
        for model in models:
            if retry_idx < len(model_stats[model]["percentages"]):
                heights.append(model_stats[model]["percentages"][retry_idx])
            else:
                heights.append(0)

        bars = ax.bar(
            x,
            heights,
            bar_width,
            bottom=bottom,
            label=f"{retry_idx} retries",
            color=colors[retry_idx] if retry_idx < len(colors) else colors[-1],
        )

        # Add percentage labels
        for i, height in enumerate(heights):
            if height > 0:
                ax.text(
                    i,
                    bottom[i] + height / 2,
                    f"{height:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                    weight="bold",
                )

        bottom += heights

    # Labels and formatting
    ax.set_ylabel("Percent of exercises completed successfully")
    ax.set_title("Model Performance by Retry Count")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.legend(title="Retry Groups")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    sample_data = {
        "model-A": [
            ("ex1", 0, 0),
            ("ex2", 1, 0),
            ("ex3", 0, 0),
            ("ex4", 2, 0),
            ("ex5", 10, 1),  # failed
        ],
        "model-B": [
            ("ex1", 0, 0),
            ("ex2", 0, 0),
            ("ex3", 1, 0),
            ("ex4", 10, 1),  # failed
        ],
    }
    visualize(sample_data)
    visualize(sample_data)
