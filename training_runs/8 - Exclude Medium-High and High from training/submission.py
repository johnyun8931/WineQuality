from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample, shuffle

ROOT = Path(__file__).resolve().parent
SEED = 42
CLASS_NAMES = ["Low", "Medium-Low", "Medium", "Medium-High", "High"]
target_count = 400
sampling_method = "none"  # Options: "none", "SMOTE", "random oversampling".
smote_k_neighbors = 5
excluded_training_classes = ["Medium-High", "High"]  # Use [] to train on all five classes.

# Edit these defaults to configure the next run.
NN_HYPERPARAMETERS = {
    "hidden_sizes": [32, 16],
    "batch_size": 64,
    "learning_rate": 0.001,
    "l2_lambda": 0.0001,
    "max_epochs": 100,
    "patience": 4,
    "min_delta": 0.0001,
    "adam_betas": [0.9, 0.999],
    "adam_eps": 1e-8,
    "seed": SEED,
    "class_weighting": "none",  # Options: "none", "sqrt_inverse_frequency".
    "dropout": 0.05,  # Fraction of hidden activations dropped during training only.
}

df = pd.read_csv(ROOT / "wine_quality.csv")

X = df.drop(columns="quality")
y = df["quality"]

# Split into 70% training, 15% validation, and 15% testing.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)

# Keep validation/test intact; remove selected classes only from training.
# Exclude them before fitting outlier thresholds or scaling as well.
original_training_counts = y_train.value_counts().to_dict()
if not set(excluded_training_classes).issubset(CLASS_NAMES):
    raise ValueError("Unknown class in excluded_training_classes")
excluded_mask = y_train.isin(excluded_training_classes)
excluded_training_counts = y_train.loc[excluded_mask].value_counts().to_dict()
X_train = X_train.loc[~excluded_mask].copy()
y_train = y_train.loc[~excluded_mask].copy()
if excluded_training_classes:
    print(f"Excluded training examples by class: {excluded_training_counts}")
    print("Validation and test retain all five classes.")

# Calculate each feature's outlier limits using training data only.
Q1 = X_train.quantile(0.25)
Q3 = X_train.quantile(0.75)
IQR = Q3 - Q1
lower_bounds = Q1 - 1.5 * IQR
upper_bounds = Q3 + 1.5 * IQR

# Remove a training row if any feature falls outside its limits.
inlier_mask = ~(
    (X_train < lower_bounds) | (X_train > upper_bounds)
).any(axis=1)
print(f"Removed {(~inlier_mask).sum()} outlier rows from training data.")

# Compare class counts before removing training outliers.
before = y_train.value_counts()
after = y_train.loc[inlier_mask].value_counts()
class_losses = pd.DataFrame({
    "before": before,
    "after": after.reindex(before.index, fill_value=0)
})
class_losses["removed"] = class_losses["before"] - class_losses["after"]
class_losses["removed_percent"] = (
    class_losses["removed"] / class_losses["before"] * 100
).round(2)
print("\nTraining examples removed by class:")
print(class_losses.to_string())

X_train = X_train.loc[inlier_mask].copy()
y_train = y_train.loc[inlier_mask].copy()

# Fit the scaler before oversampling so added rows do not change its statistics.
scaler = StandardScaler().set_output(transform="pandas")
scaler.fit(X_train)
clean_training_counts = y_train.value_counts().to_dict()

# Keep the random oversampling alternative available through sampling_method.
if sampling_method not in {"none", "random oversampling", "SMOTE"}:
    raise ValueError("sampling_method must be 'none', 'random oversampling', or 'SMOTE'")
if sampling_method == "random oversampling":
    train = X_train.copy()
    train["quality"] = y_train
    for label in ["High", "Low"]:
        class_rows = train.loc[train["quality"] == label]
        if len(class_rows) < target_count:
            extra_rows = resample(
                class_rows,
                replace=True,
                n_samples=target_count - len(class_rows),
                random_state=SEED
            )
            train = pd.concat([train, extra_rows], ignore_index=True)
    X_train = train.drop(columns="quality")
    y_train = train["quality"]

# Shuffle training features and labels together to preserve their pairing.
X_train, y_train = shuffle(X_train, y_train, random_state=SEED)

# Transform every split with the scaler fitted on cleaned training data only.
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# SMOTE uses distances, so it must run after scaling, on training data only.
if sampling_method == "SMOTE":
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(
        sampling_strategy={"High": target_count, "Low": target_count},
        random_state=SEED,
        k_neighbors=smote_k_neighbors
    )
    X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

print(f"\nTraining class counts (sampling: {sampling_method}):")
print(y_train.value_counts())


if __name__ == "__main__":
    # Importing submission from data_analysis.py prepares data without training.
    from nn_training import train_network

    train_network(
        splits={
            "train": (X_train_scaled, y_train),
            "validation": (X_val_scaled, y_val),
            "test": (X_test_scaled, y_test),
        },
        class_names=CLASS_NAMES,
        hyperparameters=NN_HYPERPARAMETERS,
        run_root=ROOT / "training_runs",
        scaler=scaler,
        metadata={
            "dataset_path": str(ROOT / "wine_quality.csv"),
            "split_fractions": [0.70, 0.15, 0.15],
            "split_seed": SEED,
            "original_training_counts": original_training_counts,
            "excluded_training_classes": excluded_training_classes,
            "excluded_training_counts": excluded_training_counts,
            "class_exclusion_stage": "after splitting, before fitting outlier thresholds/scaler",
            "outlier_rule": "any feature outside training Q1/Q3 +/- 1.5 IQR",
            "outlier_training_rows_removed": int((~inlier_mask).sum()),
            "class_losses": class_losses.to_dict(orient="index"),
            "clean_training_counts": clean_training_counts,
            "sampling_method": sampling_method,
            "sampling_targets": {"High": target_count, "Low": target_count} if sampling_method != "none" else {},
            "smote_k_neighbors": smote_k_neighbors if sampling_method == "SMOTE" else None,
            "majority_class": max(clean_training_counts, key=clean_training_counts.get),
            "scaler_fit": "clean training rows before oversampling",
        },
    )

