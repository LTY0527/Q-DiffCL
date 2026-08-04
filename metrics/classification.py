from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import label_binarize


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           probabilities: np.ndarray | None = None) -> dict[str, object]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    result: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    normal = y_true == 0
    result["far"] = float(np.mean(y_pred[normal] != 0)) if normal.any() else None
    faulty = y_true != 0
    result["mdr"] = float(np.mean(y_pred[faulty] == 0)) if faulty.any() else None
    result["auroc"] = None
    result["auprc"] = None
    if probabilities is not None and len(np.unique(y_true)) > 1:
        probabilities = np.asarray(probabilities)
        try:
            if probabilities.ndim == 1 or probabilities.shape[1] == 2:
                score = probabilities if probabilities.ndim == 1 else probabilities[:, 1]
                binary = (y_true != 0).astype(int)
                result["auroc"] = float(roc_auc_score(binary, score))
                result["auprc"] = float(average_precision_score(binary, score))
            else:
                result["auroc"] = float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro"))
                binary_targets = label_binarize(y_true, classes=np.arange(probabilities.shape[1]))
                result["auprc"] = float(average_precision_score(binary_targets, probabilities, average="macro"))
        except ValueError:
            pass
    return result


def performance_retention(degraded: float, clean: float, epsilon: float = 1e-12) -> float | None:
    return None if abs(clean) <= epsilon else degraded / clean


def drop_rate(clean: float, degraded: float, epsilon: float = 1e-12) -> float | None:
    return None if abs(clean) <= epsilon else (clean - degraded) / clean


def supcon_gain(supcon: float, ce: float) -> float:
    return supcon - ce


def select_binary_threshold(y_validation: np.ndarray, scores: np.ndarray) -> float:
    """Select a Macro-F1 threshold using validation labels only."""
    candidates = np.unique(np.r_[0.0, np.asarray(scores, dtype=float), 1.0])
    values = [f1_score(y_validation, np.asarray(scores) >= threshold, average="macro", zero_division=0) for threshold in candidates]
    return float(candidates[int(np.argmax(values))])


def detection_delay(samples: np.ndarray, predictions: np.ndarray, first_faulty_sample: float) -> float | None:
    detected = np.asarray(samples)[
        (np.asarray(samples) >= first_faulty_sample) & (np.asarray(predictions) != 0)
    ]
    return None if len(detected) == 0 else float(detected[0] - first_faulty_sample)
