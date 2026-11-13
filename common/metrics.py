"""
FootGuard AI - Evaluation Metrics
Computes multi-metric classification and regression performance reports.
Never reports accuracy alone.
"""

from typing import Dict, Any, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    num_classes: int = 2
) -> Dict[str, Any]:
    """
    Computes rigorous classification metrics.
    Supports binary and multi-class evaluations.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    
    if num_classes == 2:
        metrics.update({
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        })
        
        # AUROC & AUPRC (if probabilities are provided and both classes exist)
        if y_prob is not None:
            y_prob = np.asarray(y_prob)
            prob_pos = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] == 2 else y_prob
            if len(np.unique(y_true)) > 1:
                try:
                    metrics["auroc"] = float(roc_auc_score(y_true, prob_pos))
                    metrics["auprc"] = float(average_precision_score(y_true, prob_pos))
                except Exception:
                    metrics["auroc"] = None
                    metrics["auprc"] = None
            else:
                metrics["auroc"] = None
                metrics["auprc"] = None
    else:
        metrics.update({
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        })
        if y_prob is not None and len(np.unique(y_true)) > 1:
            try:
                metrics["auroc_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
            except Exception:
                metrics["auroc_ovr"] = None
                
    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Computes regression metrics: MAE, MSE, RMSE, R2.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2_score": float(r2)
    }

