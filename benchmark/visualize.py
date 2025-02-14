import matplotlib.pyplot as plt
import numpy as np


def visualize():
    # Data
    models = [
        "gpt-3.5-turbo-0301",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-0613",
    ]
    edit_formats = ["diff", "diff-func", "whole", "whole-func"]

    # Percentage data corresponding to (models x edit formats)
    data = np.array(
        [
            [30, 0, 54, 0],  # gpt-3.5-turbo-0301
            [18, 45, 42, 53],  # gpt-3.5-turbo-0613
            [20, 44, 41, 54],  # gpt-3.5-turbo-16k-0613
            [68, 0, 68, 0],  # gpt-4-0314
            [65, 0, 58, 68],  # gpt-4-0613
        ]
    )

    # Additional random stacks for some bars
    extra_stacks = np.array(
        [
            [10, 0, 3, 0],  # gpt-3.5-turbo-0301
            [3, 1, 2, 0],  # gpt-3.5-turbo-0613
            [10, 0, 1, 2],  # gpt-3.5-turbo-16k-0613
            [2, 0, 0, 0],  # gpt-4-0314
            [1, 3, 2, 1],  # gpt-4-0613
        ]
    )

    # Bar width and positions
    x = np.arange(len(models))
    bar_width = 0.2

    # Define colors and patterns
    colors = ["#a6dba0", "#5aae61", "#377eb8", "#b3cde3"]
    hatch_patterns = ["", "//", "", "//"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each edit format with additional small stacks
    for i, (fmt, color, hatch) in enumerate(zip(edit_formats, colors, hatch_patterns)):
        bars = ax.bar(
            x + i * bar_width - (1.5 * bar_width),
            data[:, i],
            width=bar_width,
            label=fmt,
            color=color,
            hatch=hatch,
            edgecolor="black",
        )
        # Add extra random stacks on top of the smaller stacks
        ax.bar(
            x + i * bar_width - (1.5 * bar_width),
            extra_stacks[:, i],
            width=bar_width,
            bottom=data[:, i],
            color=color,  # Use the same color as the main bar
            alpha=0.5,
            edgecolor="black",
        )
        # Add percentage labels for main bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 2,
                    f"{int(height)}%",
                    ha="center",
                    fontsize=10,
                )

    # Labels and formatting
    ax.set_ylabel("Percent of exercises completed successfully")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylim(0, 100)
    ax.set_title("GPT Code Editing")
    ax.legend(title="Edit Format")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize()
