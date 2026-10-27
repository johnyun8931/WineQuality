"""Small PyTorch classifier and timestamped experiment logging for submission.py."""

import copy
import hashlib
import json
import random
import shutil
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import matplotlib

matplotlib.use("Agg")  # Save plots without opening windows during training.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class WineQualityNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=5, dropout=0.0):
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be at least 0 and less than 1")
        layers = []
        sizes = [input_size, *hidden_sizes, output_size]
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            layer = nn.Linear(n_in, n_out)
            # He normal initialization; biases start at zero.
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))
        # Five raw logits: CrossEntropyLoss already includes log-softmax.
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        return self.network(features)


def l2_penalty(model, strength):
    # Objective = mean cross entropy + (lambda / 2) * sum(weight ** 2).
    # Biases are excluded. Adam's weight_decay stays zero to avoid counting twice.
    return 0.5 * strength * sum(
        parameter.square().sum()
        for name, parameter in model.named_parameters()
        if name.endswith("weight")
    )


@torch.no_grad()
def evaluate(model, features, labels, criterion, num_classes, weighted_criterion=None):
    model.eval()
    logits = model(features)
    probabilities = logits.softmax(dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    truth = labels.numpy()
    scores = {
        "cross_entropy": criterion(logits, labels).item(),
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(
            truth, predictions, labels=list(range(num_classes)),
            average="macro", zero_division=0,
        )),
        "weighted_f1": float(f1_score(
            truth, predictions, labels=list(range(num_classes)),
            average="weighted", zero_division=0,
        )),
        "n_samples": len(truth),
    }
    if weighted_criterion is not None:
        scores["class_weighted_cross_entropy"] = weighted_criterion(logits, labels).item()
    return scores, predictions, probabilities


def plot_with_settings(settings_text, title, figsize=(13, 7)):
    fig = plt.figure(figsize=figsize, layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=[2.1, 1])
    ax = fig.add_subplot(grid[0])
    sidebar = fig.add_subplot(grid[1])
    sidebar.axis("off")
    sidebar.text(0, 1, settings_text, va="top", fontsize=9, linespacing=1.5)
    ax.set_title(title)
    return fig, ax


def save_plots(run_dir, history, results, reports, matrices, class_names, settings, best_epoch):
    curves = [
        ("cross_entropy", "Unweighted cross entropy (L2 excluded)", "loss.png"),
        ("accuracy", "Accuracy", "accuracy.png"),
        ("macro_f1", "Macro F1", "macro_f1.png"),
    ]
    if "train_class_weighted_cross_entropy" in history.columns:
        curves.append(("class_weighted_cross_entropy", "Class-weighted cross entropy (L2 excluded)",
                       "class_weighted_loss.png"))
    for metric, label, filename in curves:
        fig, ax = plot_with_settings(settings, f"Training and validation: {label}")
        ax.plot(history["epoch"], history[f"train_{metric}"], label="Training")
        ax.plot(history["epoch"], history[f"validation_{metric}"], label="Validation")
        ax.axvline(best_epoch, color="gray", linestyle="--", label=f"Restored epoch {best_epoch}")
        ax.set(xlabel="Epoch", ylabel=label)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.savefig(run_dir / filename, dpi=150)
        plt.close(fig)

    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    fig, ax = plot_with_settings(settings, "Scores at the restored best-validation checkpoint")
    positions = np.arange(len(metrics))
    for i, split in enumerate(results):
        bars = ax.bar(positions + (i - 1) * 0.25,
                      [results[split][m] for m in metrics], width=0.25, label=split)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
    ax.set_xticks(positions, ["Accuracy", "Balanced\naccuracy", "Macro F1", "Weighted F1"])
    ax.set(ylim=(0, 1.12), ylabel="Score")
    ax.legend()
    fig.savefig(run_dir / "scores.png", dpi=150)
    plt.close(fig)

    for split in ["validation", "test"]:
        matrix = matrices[split]
        for normalize in [False, True]:
            display = matrix / matrix.sum(axis=1, keepdims=True) if normalize else matrix
            description = "fraction of true class" if normalize else "counts"
            fig, ax = plot_with_settings(settings, f"{split.title()} confusion matrix ({description})")
            im = ax.imshow(display, cmap="Blues", vmin=0, vmax=1 if normalize else None)
            fig.colorbar(im, ax=ax, fraction=0.045)
            for row in range(len(class_names)):
                for col in range(len(class_names)):
                    value = display[row, col]
                    ax.text(col, row, f"{value:.2f}" if normalize else str(value),
                            ha="center", va="center",
                            color="white" if value > display.max() / 2 else "black")
            ax.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right")
            ax.set_yticks(range(len(class_names)), class_names)
            ax.set(xlabel="Predicted quality", ylabel="True quality")
            suffix = "normalized" if normalize else "counts"
            fig.savefig(run_dir / f"{split}_confusion_{suffix}.png", dpi=150)
            plt.close(fig)

    fig, ax = plot_with_settings(settings, "Test precision, recall, and F1 by class")
    positions = np.arange(len(class_names))
    for i, metric in enumerate(["precision", "recall", "f1-score"]):
        bars = ax.bar(positions + (i - 1) * 0.25,
                      [reports["test"][name][metric] for name in class_names],
                      width=0.25, label=metric)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
    ax.set_xticks(positions, class_names, rotation=25, ha="right")
    ax.set(ylim=(0, 1.12), ylabel="Score")
    ax.legend()
    fig.savefig(run_dir / "test_per_class.png", dpi=150)
    plt.close(fig)


def train_network(splits, class_names, hyperparameters, run_root, scaler, metadata):
    """Train with validation-only early stopping, then evaluate the test set once."""
    config = dict(hyperparameters)
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)  # Efficient and repeatable for this small CPU model.
    torch.use_deterministic_algorithms(True)
    class_to_index = {name: i for i, name in enumerate(class_names)}
    tensors = {}
    for split, (features, labels) in splits.items():
        encoded = labels.map(class_to_index)
        if encoded.isna().any():
            raise ValueError(f"Unknown quality label in {split}")
        tensors[split] = (
            torch.tensor(features.to_numpy(dtype=np.float32)),
            torch.tensor(encoded.to_numpy(dtype=np.int64)),
        )

    features, labels = tensors["train"]
    weighting = config.get("class_weighting", "none")
    class_weights = None
    if weighting == "sqrt_inverse_frequency":
        # Counts come only from cleaned training rows, before any resampling.
        counts = np.array([metadata["clean_training_counts"].get(name, 0)
                           for name in class_names], dtype=np.float64)
        if (counts <= 0).any():
            raise ValueError("Every class needs training examples to calculate class weights")
        class_weights = torch.tensor(
            np.sqrt(counts.sum() / (len(class_names) * counts)), dtype=torch.float32
        )
    elif weighting != "none":
        raise ValueError("class_weighting must be 'none' or 'sqrt_inverse_frequency'")
    dropout = config.get("dropout", 0.0)
    model = WineQualityNN(features.shape[1], config["hidden_sizes"], len(class_names), dropout=dropout)
    # Keep evaluation/early stopping unweighted, as in previous runs.
    criterion = nn.CrossEntropyLoss()
    training_criterion = nn.CrossEntropyLoss(weight=class_weights)
    weighted_criterion = training_criterion if class_weights is not None else None
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"],
        betas=tuple(config["adam_betas"]), eps=config["adam_eps"], weight_decay=0,
    )
    loader = DataLoader(
        TensorDataset(features, labels), batch_size=config["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )
    started = datetime.now(ZoneInfo("America/Los_Angeles"))
    run_dir = Path(run_root) / started.strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    config.update({
        "architecture": [features.shape[1], *config["hidden_sizes"], len(class_names)],
        "activation": "ReLU (hidden); linear logits (output)",
        "loss": "class-weighted mean cross entropy" if class_weights is not None else "mean cross entropy",
        "class_weighting": weighting,
        "class_weight_formula": "sqrt(N / (num_classes * class_count))" if class_weights is not None else None,
        "class_weight_source": "clean training rows before resampling" if class_weights is not None else None,
        "class_weights": dict(zip(class_names, class_weights.tolist())) if class_weights is not None else None,
        "evaluation_loss": "unweighted mean cross entropy",
        "dropout": dropout,
        "dropout_placement": "after each hidden ReLU; disabled during evaluation",
        "regularization": "L2: lambda/2 * sum(weights squared); biases excluded",
        "optimizer": "Adam minibatch updates",
        "learning_rate_scheduler": "none (constant base learning rate)",
        "initialization": "Kaiming/He normal; zero biases",
        "early_stopping_monitor": "validation cross entropy",
        "device": "cpu",
        "torch_threads": 1,
        "class_names": class_names,
        "feature_names": list(splits["train"][0].columns),
        "split_sizes": {name: len(values[1]) for name, values in splits.items()},
        "class_counts": {
            name: values[1].value_counts().reindex(class_names, fill_value=0).to_dict()
            for name, values in splits.items()
        },
    })
    metadata = dict(metadata)
    metadata["dataset_sha256"] = hashlib.sha256(Path(metadata["dataset_path"]).read_bytes()).hexdigest()
    metadata["started_at"] = started.isoformat()
    metadata["versions"] = {
        package: version(package)
        for package in ["torch", "numpy", "pandas", "scikit-learn", "matplotlib", "joblib"]
    }
    if metadata["sampling_method"] == "SMOTE":
        metadata["versions"]["imbalanced-learn"] = version("imbalanced-learn")
    (run_dir / "hyperparameters.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "data_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    joblib.dump(scaler, run_dir / "scaler.joblib")
    for source in ["submission.py", "nn_training.py"]:
        shutil.copy2(Path(__file__).parent / source, run_dir / source)

    # Print settings after setup and immediately before the first training update.
    print(f"\nRun directory: {run_dir}")
    print("Hyperparameters (before training):")
    print(json.dumps(config, indent=2))
    print("Sampling settings:")
    print(json.dumps({key: metadata[key] for key in
                      ["sampling_method", "sampling_targets", "smote_k_neighbors"]}, indent=2))
    start_time = time.perf_counter()
    history = []
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        for batch_features, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = training_criterion(model(batch_features), batch_labels)
            objective = loss + l2_penalty(model, config["l2_lambda"])
            if not torch.isfinite(objective):
                raise RuntimeError("Nonfinite training objective")
            objective.backward()
            optimizer.step()

        # Evaluate both curves at the same end-of-epoch weights, without L2.
        train_scores, _, _ = evaluate(model, *tensors["train"], criterion, len(class_names), weighted_criterion)
        val_scores, _, _ = evaluate(model, *tensors["validation"], criterion, len(class_names), weighted_criterion)
        penalty = l2_penalty(model, config["l2_lambda"]).detach().item()
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
               "l2_penalty": penalty,
               "train_objective": train_scores.get("class_weighted_cross_entropy", train_scores["cross_entropy"]) + penalty}
        for split, scores in [("train", train_scores), ("validation", val_scores)]:
            row.update({f"{split}_{key}": value for key, value in scores.items()})
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(f"Epoch {epoch:03d} | train CE {train_scores['cross_entropy']:.4f} "
              f"| val CE {val_scores['cross_entropy']:.4f} "
              f"| val accuracy {val_scores['accuracy']:.3f} "
              f"| val macro F1 {val_scores['macro_f1']:.3f}")

        if val_scores["cross_entropy"] < best_loss - config["min_delta"]:
            best_loss = val_scores["cross_entropy"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config["patience"]:
                print(f"Early stopping: {config['patience']} epochs without sufficient validation improvement.")
                break

    model.load_state_dict(best_state)
    torch.save({
        "model_state_dict": model.state_dict(), "hyperparameters": config,
        "best_epoch": best_epoch, "class_to_index": class_to_index,
    }, run_dir / "best_model.pt")
    training_seconds = time.perf_counter() - start_time
    results, reports, matrices = {}, {}, {}
    for split, (split_features, split_labels) in tensors.items():
        scores, predictions, probabilities = evaluate(
            model, split_features, split_labels, criterion, len(class_names), weighted_criterion
        )
        results[split] = scores
        reports[split] = classification_report(
            split_labels.numpy(), predictions, labels=list(range(len(class_names))),
            target_names=class_names, output_dict=True, zero_division=0,
        )
        pd.DataFrame(reports[split]).T.to_csv(run_dir / f"{split}_classification_report.csv")
        matrices[split] = confusion_matrix(
            split_labels.numpy(), predictions, labels=list(range(len(class_names)))
        )
        pd.DataFrame(matrices[split], index=class_names, columns=class_names).to_csv(
            run_dir / f"{split}_confusion_matrix.csv", index_label="true / predicted"
        )
        prediction_table = pd.DataFrame({
            "row_index": splits[split][0].index,
            "true_quality": np.array(class_names)[split_labels.numpy()],
            "predicted_quality": np.array(class_names)[predictions],
        })
        for i, name in enumerate(class_names):
            prediction_table[f"probability_{name}"] = probabilities[:, i]
        prediction_table.to_csv(run_dir / f"{split}_predictions.csv", index=False)

    majority = class_to_index[metadata["majority_class"]]
    baseline = {}
    for split in ["validation", "test"]:
        truth = tensors[split][1].numpy()
        predicted = np.full_like(truth, majority)
        baseline[split] = {
            "predicted_class": metadata["majority_class"],
            "accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, labels=list(range(len(class_names))),
                                       average="macro", zero_division=0)),
        }
    summary = {
        "best_epoch": best_epoch, "epochs_trained": len(history),
        "stopped_early": stale_epochs >= config["patience"],
        "training_seconds": training_seconds, "scores": results,
        "majority_baseline": baseline,
        "note": (
            "Training scores use cleaned data without resampling; validation/test keep original distributions."
            if metadata["sampling_method"] == "none" else
            "Training scores use resampled data; validation/test keep original distributions."
        ),
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(results).T.to_csv(run_dir / "scores.csv", index_label="split")

    settings = "\n".join([
        f"Run: {run_dir.name}",
        "Network: " + " -> ".join(map(str, config["architecture"])),
        "Hidden activation: ReLU", "Output: 5 logits",
        f"Dropout: {dropout} after each hidden ReLU",
        "Train loss: weighted CE" if class_weights is not None else "Train loss: cross entropy",
        "Evaluation/monitor: unweighted CE",
        *(["Class weighting: sqrt inverse frequency", "Weights (Low -> High): " +
           ", ".join(f"{weight:.3f}" for weight in class_weights.tolist())]
          if class_weights is not None else ["Class weighting: none"]),
        "Initialization: He normal; zero bias", "Optimizer: Adam (minibatches)",
        f"Learning rate: {config['learning_rate']}", "LR scheduler: none",
        f"Adam betas: {config['adam_betas']}", f"Adam epsilon: {config['adam_eps']}",
        f"Batch size: {config['batch_size']}", f"L2 lambda: {config['l2_lambda']} (weights only)",
        f"Max epochs: {config['max_epochs']}", f"Patience: {config['patience']}",
        f"Minimum improvement: {config['min_delta']}", "Monitor: validation cross entropy",
        f"Best epoch: {best_epoch}; trained: {len(history)}", f"Seed: {seed}; device: CPU",
        "Split: 70/15/15, stratified", "Outliers: training only, 1.5 IQR",
        f"Sampling: {metadata['sampling_method']}",
        *([f"SMOTE neighbors: {metadata['smote_k_neighbors']}"]
          if metadata["sampling_method"] == "SMOTE" else []),
        "Targets: High=400, Low=400" if metadata["sampling_targets"] == {"High": 400, "Low": 400}
        else f"Targets: {metadata['sampling_targets']}" if metadata["sampling_targets"] else "Targets: none",
        "Scaling: StandardScaler (training only)",
        "Train/val/test: " + "/".join(str(len(splits[s][1])) for s in splits),
    ])
    save_plots(run_dir, pd.DataFrame(history), results, reports, matrices,
               class_names, settings, best_epoch)
    print(f"\nRestored epoch {best_epoch}. Final scores:")
    print(pd.DataFrame(results).T.round(4).to_string())
    print("\nTest classification report:")
    print(pd.DataFrame(reports["test"]).T.round(4).to_string())
    print(f"\nMajority baseline test accuracy: {baseline['test']['accuracy']:.4f}")
    print(f"Saved model, settings, metrics, predictions, and plots to {run_dir}")
    return run_dir
