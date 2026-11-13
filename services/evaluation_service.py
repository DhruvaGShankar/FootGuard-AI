"""
FootGuard AI - Evaluation Service
Wraps existing classification and regression metric utilities and builds
structured comparison summaries across experiments.
"""

from typing import Dict, Any, List
import numpy as np

from common.metrics import compute_classification_metrics, compute_regression_metrics


class EvaluationService:
    """Computes standardized metrics and model comparison tables."""

    @staticmethod
    def evaluate_classification(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
    ) -> Dict[str, Any]:
        return compute_classification_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob)

    @staticmethod
    def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return compute_regression_metrics(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def build_comparison_table(experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extracts comparable summary rows from experiment records, keeping real and synthetic separate."""
        rows: List[Dict[str, Any]] = []
        for exp in experiments:
            metrics = exp.get("evaluation_metrics", {})
            rows.append(
                {
                    "experiment_id": exp.get("experiment_id", "unknown"),
                    "model_name": exp.get("model_name", "unknown"),
                    "dataset_name": exp.get("dataset_name", "unknown"),
                    "dataset_origin": exp.get("dataset_origin", "unknown"),
                    "timestamp": exp.get("timestamp", ""),
                    "auroc": metrics.get("auroc"),
                    "f1_score": metrics.get("f1_score"),
                    "accuracy": metrics.get("accuracy"),
                    "mae": metrics.get("mae"),
                    "rmse": metrics.get("rmse"),
                    "r2_score": metrics.get("r2_score"),
                }
            )
        return rows

