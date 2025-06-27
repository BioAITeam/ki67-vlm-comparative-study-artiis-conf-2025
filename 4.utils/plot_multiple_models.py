import csv
import string
import sys
from pathlib import Path
import matplotlib.pyplot as plt

DEFAULT_CSVS = [
    "5.results/4.5/bcdata/ki67_results.csv",
    "5.results/gpt-4.1-mini-2025-04-14_results/bcdata/ki67_results.csv",
    "5.results/gpt-4.1-2025-04-14_results/bcdata/ki67_results.csv",
    "5.results/4o_results/bcdata/ki67_results.csv",
]

def load_data(csv_path: Path):
    y_true, y_pred = [], []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                y_true.append(float(row["true"]))
                y_pred.append(float(row["predicted"]))
            except (KeyError, ValueError):
                continue
    return y_true, y_pred

def plot_models(csv_paths, output="5.results/ki67_comparison_plot.pdf"):
    cols = len(csv_paths)
    fig, axs = plt.subplots(1, cols, figsize=(6.5 * cols, 6.5))
    if cols == 1:
        axs = [axs]

    for i, path_str in enumerate(csv_paths):
        y_true, y_pred = load_data(Path(path_str))
        ax = axs[i]

        ax.scatter(y_true, y_pred, marker="x")
        ax.plot([0, 100], [0, 100], color="gray", linewidth=1)

        ax.set_xlabel("True Ki-67 Index [%]")
        ax.set_ylabel("Predicted Ki-67 Index [%]")
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle="--", alpha=0.5)

        letter = string.ascii_uppercase[i]
        ax.set_title(letter, loc="center", fontsize=16, fontweight="bold")

    plt.tight_layout(pad=4.0)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, format="pdf")
    plt.show()
    print(f"Plot saved to: {output}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(
            "Usage:\n"
            "  python 4.utils/plot_multiple_models.py <results1.csv> [<results2.csv> ...]\n"
            "Example:\n"
            "  python 4.utils/plot_multiple_models.py \\\n"
            "    5.results/4.5/bcdata/ki67_results.csv \\\n"
            "    5.results/gpt-4.1-mini-2025-04-14_results/bcdata/ki67_results.csv \\\n"
            "    5.results/gpt-4.1-2025-04-14_results/bcdata/ki67_results.csv \\\n"
            "    5.results/4o_results/bcdata/ki67_results.csv\n"
            "No CSV paths supplied; default list will be used."
        )
        csv_list = DEFAULT_CSVS
    else:
        csv_list = sys.argv[1:]

    plot_models(csv_list)
