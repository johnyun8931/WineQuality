"""Score a saved two-class diagnostic against its original five-class test set."""

import argparse
import hashlib
import json
import shutil
import textwrap
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

if __package__:
    from .nn_training import WineQualityNN, plot_with_settings
else:
    from nn_training import WineQualityNN, plot_with_settings


ALL_CLASSES = ["Low", "Medium-Low", "Medium", "Medium-High", "High"]


def main(run_dir):
    run_dir = Path(run_dir).resolve()
    config = json.loads((run_dir / "hyperparameters.json").read_text())
    metadata = json.loads((run_dir / "data_metadata.json").read_text())
    if config["class_names"] != ["Medium-Low", "Medium"]:
        raise ValueError("Expected a saved Medium-Low vs Medium diagnostic")
    dataset_path = Path(metadata["dataset_path"])
    protected = [run_dir / "best_model.pt", run_dir / "scaler.joblib", dataset_path]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    if hashes[str(dataset_path)] != metadata["dataset_sha256"]:
        raise ValueError("Dataset differs from the dataset used to train this model")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)
    model = WineQualityNN(len(config["feature_names"]), config["hidden_sizes"], 2, dropout=config["dropout"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    scaler = joblib.load(run_dir / "scaler.joblib")
    df = pd.read_csv(dataset_path)
    test = df.loc[metadata["original_test_indices"]].copy()
    assert len(test) == metadata["original_test_size"] == 750
    assert test["quality"].value_counts().to_dict() == metadata["original_test_class_counts"]
    features = torch.tensor(scaler.transform(test[config["feature_names"]]).to_numpy(dtype=np.float32))
    with torch.no_grad():
        probabilities = model(features).softmax(1).numpy()
    predictions = np.array(config["class_names"])[probabilities.argmax(1)]
    truth = test["quality"].to_numpy()
    common = test["quality"].isin(config["class_names"]).to_numpy()
    assert not (predictions[~common] == truth[~common]).any()
    report = classification_report(truth, predictions, labels=ALL_CLASSES, output_dict=True, zero_division=0)
    matrix = confusion_matrix(truth, predictions, labels=ALL_CLASSES)
    baseline_predictions = np.repeat(metadata["majority_class"], len(truth))
    scores = {
        "n_samples": len(test),
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, labels=ALL_CLASSES, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predictions, labels=ALL_CLASSES, average="weighted", zero_division=0)),
        "cross_entropy": None,
        "cross_entropy_note": "Five-class cross entropy is not reported: the model has only two outputs and no probability outputs for three true classes.",
        "evaluated_classes": ALL_CLASSES,
        "model_output_classes": config["class_names"],
        "majority_baseline_accuracy": float(accuracy_score(truth, baseline_predictions)),
        "common_class_subset": {
            "n_samples": int(common.sum()),
            "accuracy": float(accuracy_score(truth[common], predictions[common])),
            "majority_baseline_accuracy": float(accuracy_score(truth[common], baseline_predictions[common])),
        },
        "classes_outside_model_outputs": {
            "n_samples": int((~common).sum()),
            "n_correct": 0,
        },
        "maximum_accuracy_with_these_two_outputs": float(common.mean()),
        "test_used_for_training_or_early_stopping": False,
        "retrained": False,
    }
    out = run_dir / "full_five_class_test_evaluation"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    pd.DataFrame([{key: scores[key] for key in ["n_samples", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "majority_baseline_accuracy"]}]).to_csv(out / "scores.csv", index=False)
    pd.DataFrame(report).T.to_csv(out / "classification_report.csv")
    pd.DataFrame(matrix, index=ALL_CLASSES, columns=ALL_CLASSES).to_csv(out / "confusion_matrix.csv", index_label="true / predicted")
    prediction_table = pd.DataFrame({"row_index": test.index, "true_quality": truth, "predicted_quality": predictions})
    for i, name in enumerate(config["class_names"]):
        prediction_table[f"two_class_probability_{name}"] = probabilities[:, i]
    prediction_table.to_csv(out / "predictions.csv", index=False)

    settings = "\n".join([
        textwrap.fill(f"Run: {run_dir.name}", width=43),
        "Evaluation: original full five-class test", "Model outputs: Medium-Low, Medium",
        "Network: " + " -> ".join(map(str, config["architecture"])),
        f"Dropout: {config['dropout']} (off for scoring)", "Activation: ReLU; output logits",
        "Training loss: cross entropy", f"L2 lambda: {config['l2_lambda']}",
        "Initialization: He normal; zero bias", "Optimizer: Adam",
        f"Learning rate: {config['learning_rate']}; scheduler: none",
        f"Adam betas: {config['adam_betas']}", f"Adam epsilon: {config['adam_eps']}",
        f"Batch size: {config['batch_size']}; max epochs: {config['max_epochs']}",
        f"Patience: {config['patience']}; min delta: {config['min_delta']}",
        f"Restored epoch: {checkpoint['best_epoch']}",
        f"Training seed: {config['seed']}; data seed: {metadata['split_seed']}",
        "Class weights: none; oversampling: none", "Scaler/outlier limits: training only",
        "Test size: 750; scoring only, no refit", "No five-class cross-entropy score",
    ])
    for normalized in [False, True]:
        values = matrix / matrix.sum(axis=1, keepdims=True) if normalized else matrix
        fig, ax = plot_with_settings(settings, "Full five-class test: two-output model confusion matrix")
        plot = ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalized else None)
        fig.colorbar(plot, ax=ax, fraction=0.045)
        for row in range(5):
            for col in range(5):
                value = values[row, col]
                ax.text(col, row, f"{value:.2f}" if normalized else str(value), ha="center", va="center",
                        color="white" if value > values.max() / 2 else "black")
        ax.set_xticks(range(5), ALL_CLASSES, rotation=35, ha="right")
        ax.set_yticks(range(5), ALL_CLASSES)
        ax.set(xlabel="Predicted quality (only two classes available)", ylabel="True quality")
        fig.savefig(out / ("confusion_normalized.png" if normalized else "confusion_counts.png"), dpi=150)
        plt.close(fig)
    fig, ax = plot_with_settings(settings, "Full five-class test: per-class scores")
    for i, metric in enumerate(["precision", "recall", "f1-score"]):
        bars = ax.bar(np.arange(5) + (i - 1) * .25, [report[name][metric] for name in ALL_CLASSES], width=.25, label=metric)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
    ax.set_xticks(range(5), ALL_CLASSES, rotation=25, ha="right")
    ax.set(ylim=(0, 1.1), ylabel="Score")
    ax.legend()
    fig.savefig(out / "per_class.png", dpi=150)
    plt.close(fig)
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == hashes[str(path)] for path in protected)
    (out / "verification.json").write_text(json.dumps({"saved_model_unchanged": True, "scaler_unchanged": True, "dataset_unchanged": True, "all_750_original_test_rows_scored": True, "rows_outside_two_output_classes": int((~common).sum()), "no_retraining": True}, indent=2), encoding="utf-8")
    (out / "README.md").write_text("# Full five-class test evaluation\n\nThe saved two-output diagnostic model was scored after training on all 750 original test rows, without refitting. It can predict only Medium-Low and Medium. All Low, Medium-High, and High examples are therefore misclassified. The two probability columns are the model's two-class probabilities, not a calibrated five-class distribution. Five-class cross entropy is not reported. The original two-class training/validation scores remain in the parent directory.\n", encoding="utf-8")
    source_copy = run_dir / Path(__file__).name
    if Path(__file__).resolve() != source_copy.resolve():
        shutil.copy2(Path(__file__), source_copy)
    print(json.dumps(scores, indent=2))
    print(pd.DataFrame(report).T.round(4).to_string())
    print(f"Saved separate evaluation to {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    main(parser.parse_args().run_dir)
