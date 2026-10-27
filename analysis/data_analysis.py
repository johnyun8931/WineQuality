"""Visualize the training data prepared by submission.py."""

from pathlib import Path
import sys

# Also support direct execution: python analysis/data_analysis.py.
ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))
PLOT_DIR = Path(__file__).resolve().parent / "plots"

import seaborn as sns
import matplotlib.pyplot as plt


def main():
    from scripts.submission import df, X_train_scaled, y_train

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # print(df.describe())

    # Plot and save the scaled training features, replacing the previous box graph.
    numeric_df = X_train_scaled.select_dtypes(include="number")
    fig, axs = plt.subplots(len(numeric_df.columns), 1, figsize=(7, 18), dpi=95, squeeze=False)
    for ax, col in zip(axs.flat, numeric_df.columns):
        ax.boxplot(numeric_df[col].dropna(), orientation="horizontal")
        ax.set_ylabel(col)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "feature graph.png", bbox_inches="tight")

    # Pearson correlations between the cleaned training features.
    # The categorical quality labels are excluded from this numeric analysis.
    correlation_matrix = X_train_scaled.corr(method="pearson")
    fig_corr, ax_corr = plt.subplots(figsize=(12, 10), dpi=120)
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson correlation"},
        ax=ax_corr,
    )
    ax_corr.set_title("Feature Correlations — Cleaned Training Data")
    plt.setp(ax_corr.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax_corr.get_yticklabels(), rotation=0)
    fig_corr.tight_layout()
    fig_corr.savefig(PLOT_DIR / "correlation_heatmap.png", bbox_inches="tight")

    # Show the class distribution in the cleaned training target.
    quality_order = ["Low", "Medium-Low", "Medium", "Medium-High", "High"]
    target_counts = y_train.value_counts().reindex(quality_order, fill_value=0)
    fig_target, ax_target = plt.subplots(figsize=(9, 5), dpi=120)
    sns.countplot(x=y_train, order=quality_order, color="steelblue", ax=ax_target)
    labels = [
        f"{count:,}\n({count / len(y_train):.1%})"
        for count in target_counts
    ]
    ax_target.bar_label(ax_target.containers[0], labels=labels, padding=4)
    ax_target.set_title("Wine Quality Distribution - Cleaned Training Data")
    ax_target.set_xlabel("Wine quality")
    ax_target.set_ylabel("Number of samples")
    ax_target.set_ylim(0, target_counts.max() * 1.20)
    fig_target.tight_layout()
    fig_target.savefig(PLOT_DIR / "target_distribution.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
