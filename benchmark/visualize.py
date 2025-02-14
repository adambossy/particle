import argparse
import csv
import os
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_results(filepath: str) -> Dict[str, List[Tuple[str, int, int]]]:
    # Extract model name from parent directory
    dirname = os.path.basename(os.path.dirname(filepath))
    # Split by underscore and take the first part (model name)
    model_name = dirname.split("_")[0]

    results = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            exercise = row["Exercise Name"]
            attempts = int(row["Num Attempts"])
            status = int(row["Return Code"])
            results.append((exercise, attempts, status))

    return {model_name: results}


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
    # Colors from team activity visualization gradient
    colors = [
        "#290958",  # darkest purple
        "#4D1B8C",  # deep purple
        "#8034B8",  # medium purple
        "#B84B8C",  # purple-pink
        "#E65F8B",  # dark pink
        "#EA7369",  # salmon pink
        "#F0988F",  # light coral
        "#F5B4B2",  # pale coral
        "#FAD1D6",  # light pink
        "#FFE9E9",  # lightest pink
    ]  # Purple to pink gradient

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
    ax.set_ylabel("Percent of exercises translated successfully")
    ax.set_title("Model Performance by Retry Count")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0, ha="center")
    ax.set_ylim(0, 100)
    ax.legend(title="Retry Groups")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model results")
    parser.add_argument("results_file", help="Path to results.txt file")
    args = parser.parse_args()

    results = load_results(args.results_file)
    visualize(results)
    visualize(results)
