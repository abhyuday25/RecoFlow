from __future__ import annotations

from math import sqrt
from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true_arr = np.array(list(y_true), dtype=np.float32)
    pred_arr = np.array(list(y_pred), dtype=np.float32)
    if len(true_arr) == 0:
        return 0.0
    return float(sqrt(np.mean((true_arr - pred_arr) ** 2)))


def evaluate_rmse(test_df: pd.DataFrame, predictor: Callable[[int, int], float]) -> float:
    predictions = [predictor(int(row.userId), int(row.movieId)) for row in test_df.itertuples(index=False)]
    return rmse(test_df["rating"].tolist(), predictions)


def precision_at_k(recommendations: Dict[int, List[int]], ground_truth: Dict[int, List[int]], k: int = 10) -> float:
    scores = []
    for user_id, recs in recommendations.items():
        top_k = recs[:k]
        relevant = set(ground_truth.get(user_id, []))
        if not top_k:
            continue
        hits = sum(1 for item in top_k if item in relevant)
        scores.append(hits / k)
    return float(np.mean(scores)) if scores else 0.0


def recall_at_k(recommendations: Dict[int, List[int]], ground_truth: Dict[int, List[int]], k: int = 10) -> float:
    scores = []
    for user_id, recs in recommendations.items():
        relevant = set(ground_truth.get(user_id, []))
        if not relevant:
            continue
        top_k = recs[:k]
        hits = sum(1 for item in top_k if item in relevant)
        scores.append(hits / len(relevant))
    return float(np.mean(scores)) if scores else 0.0
