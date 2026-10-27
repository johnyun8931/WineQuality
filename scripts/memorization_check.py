"""Check whether the NN can memorize 64 balanced training examples."""

import hashlib
import json
import random
import shutil
import textwrap
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__:
    from .nn_training import WineQualityNN, create_run_directory, evaluate, plot_with_settings
else:
    from nn_training import WineQualityNN, create_run_directory, evaluate, plot_with_settings
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["Medium-Low", "Medium"]
DATA_SEED = 42
SUBSET_SEED = 42
HYPERPARAMETERS = {
    "hidden_sizes": [64, 32],
    "batch_size": 64,
    "learning_rate": 0.001,
    "adam_betas": [0.9, 0.999],
    "adam_eps": 1e-8,
    "seed": 43,
    "dropout": 0.0,
    "l2_lambda": 0.0,
    "class_weighting": "none",
    "examples_per_class": 32,
    "max_epochs": 2000,
    "patience": None,
    "early_stopping_monitor": None,
    "target_training_accuracy": 1.0,
    "target_training_cross_entropy": 0.01,
    "target_consecutive_epochs": 5,
}


def main():
    config = dict(HYPERPARAMETERS)
    dataset_path = ROOT / "wine_quality.csv"
    original_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    df = pd.read_csv(dataset_path)
    X, y = df.drop(columns="quality"), df["quality"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=DATA_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=DATA_SEED, stratify=y_temp
    )
    selected = y_train.isin(CLASS_NAMES)
    pool = X_train.loc[selected].copy()
    pool_labels = y_train.loc[selected].copy()
    q1, q3 = pool.quantile(0.25), pool.quantile(0.75)
    iqr = q3 - q1
    inliers = ~((pool < q1 - 1.5 * iqr) | (pool > q3 + 1.5 * iqr)).any(axis=1)
    pool, pool_labels = pool.loc[inliers], pool_labels.loc[inliers]

    # Balanced selection from original training rows, without replacement.
    indices = []
    for name in CLASS_NAMES:
        indices.extend(pool_labels.loc[pool_labels == name].sample(
            n=config["examples_per_class"], replace=False, random_state=SUBSET_SEED
        ).index.tolist())
    X_small, y_small = shuffle(pool.loc[indices], pool_labels.loc[indices], random_state=SUBSET_SEED)
    scaler = StandardScaler().set_output(transform="pandas").fit(X_small)
    X_scaled = scaler.transform(X_small)
    class_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}
    features = torch.tensor(X_scaled.to_numpy(dtype=np.float32))
    labels = torch.tensor(y_small.map(class_to_index).to_numpy(dtype=np.int64))

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = WineQualityNN(features.shape[1], config["hidden_sizes"], 2, dropout=0.0)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], betas=tuple(config["adam_betas"]),
        eps=config["adam_eps"], weight_decay=0.0,
    )
    loader = DataLoader(
        TensorDataset(features, labels), batch_size=config["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"]), num_workers=0,
    )
    started = datetime.now(ZoneInfo("America/Los_Angeles"))
    metadata = {
        "task_name": "training-subset memorization check",
        "experiment_description": "Memorize 64 balanced samples without dropout or L2",
        "started_at": started.isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": original_hash,
        "split_seed": DATA_SEED,
        "subset_seed": SUBSET_SEED,
        "sampling_method": "balanced subset without replacement",
        "eligible_training_counts": pool_labels.value_counts().to_dict(),
        "training_counts": y_small.value_counts().to_dict(),
        "training_row_indices": X_small.index.tolist(),
        "original_validation_indices": X_val.index.tolist(),
        "original_test_indices": X_test.index.tolist(),
        "original_validation_size": len(X_val),
        "original_test_size": len(X_test),
        "validation_evaluated": False,
        "test_evaluated": False,
        "outlier_rule": "1.5 IQR on original two-class training pool, before subset selection",
        "outlier_pool_rows_removed": int((~inliers).sum()),
        "scaler_fit": "64 selected training examples only",
        "versions": {package: version(package) for package in ["torch", "numpy", "pandas", "scikit-learn", "matplotlib", "joblib"]},
    }
    config.update({
        "architecture": [features.shape[1], *config["hidden_sizes"], 2],
        "class_names": CLASS_NAMES,
        "feature_names": list(X_small.columns),
        "training_subset_size": len(X_small),
        "activation": "ReLU hidden; linear output logits",
        "loss": "unweighted cross entropy; no regularization",
        "initialization": "He normal; zero biases",
        "optimizer": "Adam", "weight_decay": 0.0,
        "learning_rate_scheduler": "none", "device": "cpu", "torch_threads": 1,
        "stopping_rule": "training targets satisfied for five consecutive epochs, or epoch limit",
    })
    run_dir = create_run_directory(ROOT / "training_runs", config, metadata)
    (run_dir / "hyperparameters.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "data_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    joblib.dump(scaler, run_dir / "scaler.joblib")
    pd.concat([X_small, y_small.rename("quality")], axis=1).to_csv(run_dir / "training_subset.csv", index_label="row_index")
    for source in ["memorization_check.py", "nn_training.py"]:
        shutil.copy2(Path(__file__).resolve().parent / source, run_dir / source)

    print(f"\nRun directory: {run_dir}")
    print("Training only: 32 Medium-Low and 32 Medium examples.")
    print("No dropout, L2, class weights, oversampling, or validation-based stopping.")
    print("Hyperparameters (before training):")
    print(json.dumps(config, indent=2))
    history = []
    consecutive = 0
    first_perfect_epoch = None
    clock_start = time.perf_counter()
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        for batch_features, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features), batch_labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
        scores, predictions, probabilities = evaluate(model, features, labels, criterion, 2)
        if scores["accuracy"] == 1.0 and first_perfect_epoch is None:
            first_perfect_epoch = epoch
        target_met = (scores["accuracy"] >= config["target_training_accuracy"] and
                      scores["cross_entropy"] <= config["target_training_cross_entropy"])
        consecutive = consecutive + 1 if target_met else 0
        history.append({"epoch": epoch, **scores, "learning_rate": config["learning_rate"], "target_consecutive_epochs": consecutive})
        if epoch == 1 or epoch % 25 == 0 or consecutive >= config["target_consecutive_epochs"]:
            print(f"Epoch {epoch:04d} | training loss {scores['cross_entropy']:.6f} | accuracy {scores['accuracy']:.2%} | consecutive target epochs {consecutive}")
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if consecutive >= config["target_consecutive_epochs"]:
            break

    success = consecutive >= config["target_consecutive_epochs"]
    result = {
        "memorization_success": success,
        "epochs_trained": epoch,
        "first_perfect_accuracy_epoch": first_perfect_epoch,
        "target_consecutive_epochs_reached": consecutive,
        "training_seconds": time.perf_counter() - clock_start,
        "scores": {"train": scores},
        "training_majority_baseline_accuracy": 0.5,
        "validation_evaluated": False, "test_evaluated": False,
        "interpretation": "Checks ability to fit training examples only; does not establish generalization or feature-label signal on unseen data.",
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    pd.DataFrame([scores], index=["train"]).to_csv(run_dir / "scores.csv", index_label="split")
    torch.save({"model_state_dict": model.state_dict(), "hyperparameters": config,
                "epoch": epoch, "class_to_index": class_to_index}, run_dir / "memorized_model.pt")
    report = classification_report(labels.numpy(), predictions, labels=[0, 1], target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(run_dir / "train_classification_report.csv")
    matrix = confusion_matrix(labels.numpy(), predictions, labels=[0, 1])
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(run_dir / "train_confusion_matrix.csv", index_label="true / predicted")
    prediction_table = pd.DataFrame({"row_index": X_small.index, "true_quality": y_small.to_numpy(),
                                     "predicted_quality": np.array(CLASS_NAMES)[predictions]})
    for i, name in enumerate(CLASS_NAMES):
        prediction_table[f"probability_{name}"] = probabilities[:, i]
    prediction_table.to_csv(run_dir / "train_predictions.csv", index=False)

    settings = "\n".join([
        textwrap.fill(f"Run: {run_dir.name}", width=42),
        "Purpose: memorize training examples", "Network: 11 -> 64 -> 32 -> 2",
        "Training subset: 64 (32 per class)", "ReLU hidden; linear output logits",
        "Dropout: 0; L2 lambda: 0", "Adam weight decay: 0", "Loss: unweighted cross entropy",
        "Initialization: He normal; zero bias", "Optimizer: Adam; learning rate: 0.001",
        "Adam betas: [0.9, 0.999]; eps: 1e-8", "Batch size: 64; max epochs: 2000",
        "LR scheduler: none; CPU threads: 1", "Validation early stopping: disabled",
        "Target: 100% accuracy and CE <= 0.01", "Require target for 5 consecutive epochs",
        f"Epochs trained: {epoch}; success: {success}", f"First 100% accuracy epoch: {first_perfect_epoch}",
        "Training seed: 43; subset seed: 42", "Source split seed: 42 (70/15/15)",
        "Class weights: none; no oversampling", "Scaler: selected training subset only",
        "Validation/test: not evaluated", "Training fit only; no generalization claim",
    ])
    curves = [("cross_entropy", "Training cross entropy", "loss.png", 0.01),
              ("accuracy", "Training accuracy", "accuracy.png", 1.0)]
    table = pd.DataFrame(history)
    for metric, title, filename, target in curves:
        fig, ax = plot_with_settings(settings, title + " - 64-example memorization check")
        ax.plot(table.epoch, table[metric], label="64 training examples")
        ax.axhline(target, color="gray", linestyle="--", label="Target")
        ax.set(xlabel="Epoch", ylabel=title)
        if metric == "accuracy":
            ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25); ax.legend()
        fig.savefig(run_dir / filename, dpi=150); plt.close(fig)
    fig, ax = plot_with_settings(settings, "Training-only confusion matrix - memorization check")
    ax.imshow(matrix, cmap="Blues", vmin=0)
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > matrix.max() / 2 else "black")
    ax.set_xticks(range(2), CLASS_NAMES); ax.set_yticks(range(2), CLASS_NAMES)
    ax.set(xlabel="Predicted quality", ylabel="True quality")
    fig.savefig(run_dir / "train_confusion_counts.png", dpi=150); plt.close(fig)
    (run_dir / "README.md").write_text(
        "# Training-subset memorization check\n\n"
        "This run uses 64 original training examples, balanced 32 per class, and no dropout or L2. "
        "Validation-based early stopping is disabled. Training stops at 100% accuracy with cross entropy "
        "at most 0.01 for five consecutive epochs, or after 2,000 epochs. "
        "The original validation and test sets are not transformed or evaluated. "
        "Perfect training performance confirms the network can fit these training examples; "
        "it does not show that predictions generalize. The main experiment defaults are unchanged.\n",
        encoding="utf-8",
    )
    assert hashlib.sha256(dataset_path.read_bytes()).hexdigest() == original_hash
    print("\nMemorization check complete:")
    print(json.dumps(result, indent=2))
    print(f"Saved training-only artifacts to {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
