"""
evaluation_experiments.py

Runs complete evaluation pipeline.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.visualization import (
    load_all_results,
    plot_f1,
    plot_accuracy,
    plot_precision_recall,
    plot_all_metrics,
)

from src.evaluation.model_comparison import (
    generate_full_comparison,
    save_comparison_table
)


def run_evaluation():
    print("Loading results...")

    df = load_all_results()

    print("\nSample:\n", df.head())

    print("\nGenerating plots...")

    plot_f1(df)
    plot_accuracy(df)
    plot_precision_recall(df)
    plot_all_metrics(df)

    print("\nGenerating comparison table...")

    table = generate_full_comparison(df)
    print(table)

    save_comparison_table(table)


if __name__ == "__main__":
    run_evaluation()