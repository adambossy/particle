import argparse
import csv
import os
from collections import Counter
from pprint import pformat
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def combine_results(
    results: Dict[str, List[Tuple[str, int, int]]]
) -> List[Tuple[str, int, int]]:
    """
    Combine results across models by taking the result with the lowest retry count for each exercise.

    Args:
        results: Dictionary mapping model names to lists of (exercise, attempts, status) tuples

    Returns:
        List of (exercise, attempts, status) tuples with the best result for each exercise
    """
    # Create a dictionary to track the best result for each exercise
    best_results: Dict[str, Tuple[int, int]] = {}

    # Iterate through all models and their results
    for model, exercises in results.items():
        for exercise, attempts, status in exercises:
            print(
                f"Evaluating exercise: {exercise}, attempts: {attempts}, status: {status}"
            )
            # If we haven't seen this exercise before, or if this result has fewer attempts
            # or if it has the same attempts but succeeded (status 0) where the previous one failed
            if exercise not in best_results or (
                attempts < best_results[exercise][0] and status == 0
            ):
                best_results[exercise] = (attempts, status)

    # Convert the dictionary back to a list of tuples
    combined_results = [
        (exercise, attempts, status)
        for exercise, (attempts, status) in best_results.items()
    ]

    print("Final combined_results:", pformat(combined_results))

    return combined_results


def visualize_bar_chart(
    results: Dict[str, List[Tuple[str, int, int]]], output_path: str
) -> None:
    # Process data
    model_stats = {}

    # Add combined results
    combined_results = combine_results(results)
    results_with_combined = results.copy()
    results_with_combined["combined"] = combined_results

    for model, exercises in results_with_combined.items():
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
            "success_percentage": len(successful)
            / total_count
            * 100,  # Calculate success percentage
        }

        # Print statistics
        print(f"\nModel: {model}")
        for retries, count in sorted_retries:
            print(f"Retry group {retries}: {count} exercises")
        print(f"Total exercises: {total_count}")

    # Plotting
    models = list(results_with_combined.keys())
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

    # Add success percentage labels above each bar
    for i, model in enumerate(models):
        success_percentage = model_stats[model]["success_percentage"]
        ax.text(
            x[i],
            bottom[i] + 2,  # Position above the top of the bar
            f"{success_percentage:.1f}%",
            ha="center",
            va="bottom",
            color="black",
            weight="bold",
        )

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
    # plt.show()


def visualize_table(
    results: Dict[str, List[Tuple[str, int, int]]], output_path: str
) -> None:
    """
    Create a table visualization with exercises as rows and models as columns.
    Each cell contains the retry count for that exercise and model.

    Args:
        results: Dictionary mapping model names to lists of (exercise, attempts, status) tuples
        output_path: Path to save the visualization
    """
    # Create a dictionary to store retry counts for each exercise and model
    exercise_model_retries: Dict[str, Dict[str, Optional[Tuple[int, int]]]] = {}

    # Get all unique exercises across all models
    all_exercises = set()
    for exercises in results.values():
        all_exercises.update(ex[0] for ex in exercises)

    # Initialize the dictionary with None values
    for exercise in all_exercises:
        exercise_model_retries[exercise] = {model: None for model in results.keys()}

    # Fill in the retry counts and status codes
    for model, exercises in results.items():
        for exercise, attempts, status in exercises:
            exercise_model_retries[exercise][model] = (attempts, status)

    # Add combined results
    combined_results = combine_results(results)
    combined_dict = {ex[0]: ((ex[1], ex[2])) for ex in combined_results}

    for exercise in all_exercises:
        exercise_model_retries[exercise]["combined"] = combined_dict.get(exercise)

    # Convert to DataFrame for easier visualization
    df = pd.DataFrame.from_dict(exercise_model_retries, orient="index")

    # Sort by exercise name
    df = df.sort_index()

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, max(8, len(all_exercises) * 0.3)))
    ax.axis("off")

    # Create the table
    table = ax.table(
        cellText=df.applymap(
            lambda x: f"{x[0]} ({x[1]})" if x is not None else "N/A"
        ).values,
        rowLabels=df.index,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Color cells based on retry count
    for i in range(len(df)):
        for j in range(len(df.columns)):
            cell = table[(i + 1, j)]
            value = df.iloc[i, j]
            if value is not None:
                # Color gradient from green (0 retries) to red (max retries)
                max_retries = 10  # Adjust as needed
                if value[0] <= max_retries:
                    # Green to red gradient
                    r = min(1.0, value[0] / max_retries)
                    g = max(0.0, 1.0 - value[0] / max_retries)
                    cell.set_facecolor((r, g, 0.0))

                    # Set text color to white for better visibility on darker backgrounds
                    if value[0] >= max_retries / 2:
                        cell.get_text().set_color("white")

    plt.title("Exercise Retry Counts by Model")
    plt.tight_layout()

    # Save the table visualization
    table_output_path = output_path.replace(".png", "_table.png")
    plt.savefig(table_output_path, dpi=300, bbox_inches="tight")
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
    visualize_bar_chart(results, output_path)
    visualize_table(results, output_path)
