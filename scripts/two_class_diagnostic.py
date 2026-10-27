"""Separate Medium-Low vs Medium experiment; the original test set is held out."""

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

if __package__:
    from .nn_training import train_network
else:
    from nn_training import train_network


ROOT = Path(__file__).resolve().parents[1]
DATA_SEED = 42
CLASS_NAMES = ["Medium-Low", "Medium"]

# Match the current five-class NN settings; only the diagnostic task changes.
HYPERPARAMETERS = {
    "hidden_sizes": [64, 32],
    "batch_size": 64,
    "learning_rate": 0.001,
    "l2_lambda": 0.0001,
    "max_epochs": 100,
    "patience": 2,
    "min_delta": 0.0001,
    "adam_betas": [0.9, 0.999],
    "adam_eps": 1e-8,
    "seed": 43,
    "class_weighting": "none",
    "dropout": 0.05,
}


def main():
    dataset_path = ROOT / "wine_quality.csv"
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    df = pd.read_csv(dataset_path)
    X, y = df.drop(columns="quality"), df["quality"]

    # Reproduce the original FIVE-class split before choosing diagnostic classes.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=DATA_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=DATA_SEED, stratify=y_temp
    )
    original_training_counts = y_train.value_counts().to_dict()
    original_validation_counts = y_val.value_counts().to_dict()
    held_out_test_indices = X_test.index.tolist()

    # Only training and validation are filtered. X_test/y_test stay five-class.
    train_mask = y_train.isin(CLASS_NAMES)
    validation_mask = y_val.isin(CLASS_NAMES)
    excluded_counts = y_train.loc[~train_mask].value_counts().to_dict()
    X_train, y_train = X_train.loc[train_mask].copy(), y_train.loc[train_mask].copy()
    X_val, y_val = X_val.loc[validation_mask].copy(), y_val.loc[validation_mask].copy()

    # Learn preprocessing from the two-class training data alone.
    q1, q3 = X_train.quantile(0.25), X_train.quantile(0.75)
    iqr = q3 - q1
    inliers = ~((X_train < q1 - 1.5 * iqr) | (X_train > q3 + 1.5 * iqr)).any(axis=1)
    before = y_train.value_counts()
    after = y_train.loc[inliers].value_counts().reindex(before.index, fill_value=0)
    losses = pd.DataFrame({"before": before, "after": after})
    losses["removed"] = losses["before"] - losses["after"]
    losses["removed_percent"] = (100 * losses["removed"] / losses["before"]).round(2)
    X_train, y_train = X_train.loc[inliers].copy(), y_train.loc[inliers].copy()
    scaler = StandardScaler().set_output(transform="pandas").fit(X_train)
    counts = y_train.value_counts().to_dict()
    X_train, y_train = shuffle(X_train, y_train, random_state=DATA_SEED)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    # No test transformation, prediction, or score is requested or performed.

    print("\nTwo-class diagnostic: Medium-Low vs Medium")
    print("Training examples removed as outliers:")
    print(losses.to_string())
    print("\nTraining class counts:")
    print(y_train.value_counts())
    print("\nValidation class counts:")
    print(y_val.value_counts())
    print(f"Original five-class test: {len(X_test)} rows held out, not evaluated.")

    run_dir = train_network(
        splits={"train": (X_train_scaled, y_train), "validation": (X_val_scaled, y_val)},
        class_names=CLASS_NAMES,
        hyperparameters=HYPERPARAMETERS,
        run_root=ROOT / "training_runs",
        scaler=scaler,
        metadata={
            "task_name": "two-class diagnostic",
            "experiment_description": "Two-class diagnostic Medium-Low vs Medium",
            "source_files": ["two_class_diagnostic.py", "nn_training.py", "submission.py"],
            "dataset_path": str(dataset_path),
            "split_fractions": [0.70, 0.15, 0.15],
            "split_seed": DATA_SEED,
            "original_training_counts": original_training_counts,
            "original_validation_counts": original_validation_counts,
            "excluded_training_classes": ["Low", "Medium-High", "High"],
            "excluded_training_counts": excluded_counts,
            "validation_classes": CLASS_NAMES,
            "class_exclusion_stage": "training/validation filtered after original five-class split; before preprocessing",
            "outlier_rule": "any feature outside two-class training Q1/Q3 +/- 1.5 IQR",
            "outlier_training_rows_removed": int((~inliers).sum()),
            "class_losses": losses.to_dict(orient="index"),
            "clean_training_counts": counts,
            "sampling_method": "none",
            "sampling_targets": {},
            "smote_k_neighbors": None,
            "majority_class": max(counts, key=counts.get),
            "scaler_fit": "clean two-class training rows only",
            "original_test_size": len(X_test),
            "original_test_indices": held_out_test_indices,
            "original_test_class_counts": y_test.value_counts().to_dict(),
            "test_evaluated": False,
            "evaluation_note": "Two-class training and validation only. Original five-class test is untouched and not evaluated; these are not five-class test scores.",
        },
    )
    assert hashlib.sha256(dataset_path.read_bytes()).hexdigest() == dataset_hash
    return run_dir


if __name__ == "__main__":
    main()
