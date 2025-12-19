"""
utils.py
Common utility functions used across different methods.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# Reproducibility
def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass



# Dataset utilities
def count_images(data_dir):
    """
    Count number of images per class in train/validation/test folders.
    """
    stats = {}
    for split in ["train", "validation", "test"]:
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            continue

        stats[split] = {}
        for cls in os.listdir(split_path):
            cls_path = os.path.join(split_path, cls)
            stats[split][cls] = len(os.listdir(cls_path))

    return stats


# Evaluation utilities
def compute_metrics(y_true, y_pred, class_names=("healthy", "mci")):
    """
    Compute classification metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1_score": f1_score(y_true, y_pred, average="weighted")
    }

    report = classification_report(
        y_true, y_pred, target_names=class_names
    )

    return metrics, report


def plot_confusion_matrix(y_true, y_pred, class_names=("healthy", "mci")):
    """
    Plot confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j],
                     ha="center", va="center", color="black")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()
