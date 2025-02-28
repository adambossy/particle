import argparse
import csv
import os
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_results(filepaths: List[str]) -> Dict[str, List[Tuple[str, int, int]]]:
    results = {}
    for filepath in filepaths:
        # Get the directory containing the file
        dirname = os.path.dirname(filepath)
        # Get the parent directory of that directory
        parent_dir = os.path.basename(os.path.dirname(dirname))
        # Split by underscore and take the first part (model name)
        model_name = parent_dir.split("_")[0]

        if model_name not in results:
            results[model_name] = []

        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                exercise = row["Exercise Name"]
                attempts = int(row["Num Attempts"])
                status = int(row["Return Code"])
                results[model_name].append((exercise, attempts, status))

    return results


def visualize(results: Dict[str, List[Tuple[str, int, int]]], output_path: str) -> None:
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
    # Save the plot instead of showing it
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model results")
    parser.add_argument(
        "results_dir", help="Path to the directory containing model subdirectories"
    )
    args = parser.parse_args()

    # Walk through each subdirectory and collect results.txt files
    results_files = []
    for root, dirs, files in os.walk(args.results_dir):
        for dir in dirs:
            results_file = os.path.join(root, dir, "results/", "results.txt")
            print(
                f"Found results file (exists?): {results_file} {os.path.exists(results_file)}"
            )
            if os.path.exists(results_file):
                results_files.append(results_file)

    # Save the visualization in the results directory
    output_path = os.path.join(args.results_dir, "visualization.png")

    results = load_results(results_files)
    visualize(results, output_path)
