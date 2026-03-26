"""
ExDA - Main Usage Example

This script demonstrates how to use ExDA for data augmentation
in Cross-Project Defect Prediction (CPDP) scenarios.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

from exda import ExDA
from exda.data_loader import load_dataset


def evaluate_model(X_train, y_train, X_test, y_test):
    """Train and evaluate a classifier."""
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return {
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }


def main():
    # Example: Load AEEEM dataset (EQ as source, JDT as target)
    dataset_path = "./datasets/AEEEM/EQ.csv"
    df = pd.read_csv(dataset_path)

    # Assume last column is the target (class)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("=" * 50)
    print("Original Data:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Positive rate: {np.mean(y_train):.2%}")
    print("=" * 50)

    # Baseline evaluation (without ExDA)
    baseline_results = evaluate_model(X_train, y_train, X_test, y_test)
    print(f"\nBaseline (no augmentation):")
    print(f"  Precision: {baseline_results['precision']:.4f}")
    print(f"  Recall: {baseline_results['recall']:.4f}")
    print(f"  F1: {baseline_results['f1']:.4f}")

    # ExDA augmentation
    print("\n" + "=" * 50)
    print("Training ExDA...")
    exda = ExDA(
        lambda_param=0.7,         # Hybrid importance weight
        k_top_features=0.25,      # Top 25% features as semantic anchors
        latent_dim=32,            # VAE latent dimension
        hidden_dim=64,            # VAE hidden dimension
        lr=1e-3,                  # Learning rate
        epochs=50,                # Training epochs
        batch_size=16,            # Batch size
        augmentation_percentage=0.3,  # 30% augmentation
        target_ratio=0.5          # Target class ratio
    )

    exda.fit(X_train, y_train)
    X_aug, y_aug = exda.augment(X_train, y_train)

    print(f"\nAugmented Data:")
    print(f"  Training samples: {len(X_aug)}")
    print(f"  Positive rate: {np.mean(y_aug):.2%}")
    print("=" * 50)

    # Evaluation with ExDA
    exda_results = evaluate_model(X_aug, y_aug, X_test, y_test)
    print(f"\nWith ExDA augmentation:")
    print(f"  Precision: {exda_results['precision']:.4f}")
    print(f"  Recall: {exda_results['recall']:.4f}")
    print(f"  F1: {exda_results['f1']:.4f}")

    # Improvement
    print("\n" + "=" * 50)
    print("Improvement:")
    print(f"  Recall: +{exda_results['recall'] - baseline_results['recall']:.4f}")
    print(f"  F1: +{exda_results['f1'] - baseline_results['f1']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()