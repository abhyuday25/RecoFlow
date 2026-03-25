from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
import torch

from app.config import ARTIFACTS_DIR, DEFAULT_TOP_K, MOVIES_PATH, RATINGS_PATH
from app.data.dataset import (
    build_ground_truth,
    encode_dataset,
    load_movielens_data,
    train_test_split_ratings,
)
from app.models.collaborative_filtering import CollaborativeFilteringRecommender, Recommendation
from app.models.matrix_factorization import SVDRecommender
from app.models.neural_cf import NeuralCFRecommender
from app.utils.artifacts import ensure_dir, load_json, save_json
from app.utils.metrics import evaluate_rmse, precision_at_k, recall_at_k


@dataclass
class EvaluationSummary:
    model_name: str
    rmse: float
    precision_at_k: float
    recall_at_k: float


class RecommendationService:
    def __init__(self, strategy: str = "hybrid") -> None:
        self.strategy = strategy
        self.ratings, self.movies = load_movielens_data(str(RATINGS_PATH), str(MOVIES_PATH))
        self.encoded = encode_dataset(self.ratings, self.movies)
        self.cf = CollaborativeFilteringRecommender(self.encoded)
        self.svd = SVDRecommender(self.encoded)
        self.ncf = NeuralCFRecommender(self.encoded)

    def train_and_evaluate(self, top_k: int = DEFAULT_TOP_K) -> List[EvaluationSummary]:
        train_df, test_df = train_test_split_ratings(self.ratings)
        train_encoded = encode_dataset(train_df, self.movies)

        cf = CollaborativeFilteringRecommender(train_encoded)
        svd = SVDRecommender(train_encoded)
        svd.fit()
        ncf = NeuralCFRecommender(train_encoded)
        ncf.fit()

        ground_truth = build_ground_truth(test_df)
        user_ids = sorted(test_df["userId"].unique())

        def collect_rankings(recommender, recommend_kwargs: Dict | None = None) -> Dict[int, List[int]]:
            rankings: Dict[int, List[int]] = {}
            recommend_kwargs = recommend_kwargs or {}
            for user_id in user_ids:
                rankings[user_id] = [
                    rec.movie_id for rec in recommender.recommend(int(user_id), top_n=top_k, **recommend_kwargs)
                ]
            return rankings

        evaluations = [
            EvaluationSummary(
                model_name="collaborative_filtering",
                rmse=evaluate_rmse(test_df, lambda u, i: cf.predict_rating(u, i, strategy=self.strategy)),
                precision_at_k=precision_at_k(
                    collect_rankings(cf, {"strategy": self.strategy}), ground_truth, k=top_k
                ),
                recall_at_k=recall_at_k(collect_rankings(cf, {"strategy": self.strategy}), ground_truth, k=top_k),
            ),
            EvaluationSummary(
                model_name="matrix_factorization_svd",
                rmse=evaluate_rmse(test_df, svd.predict_rating),
                precision_at_k=precision_at_k(collect_rankings(svd), ground_truth, k=top_k),
                recall_at_k=recall_at_k(collect_rankings(svd), ground_truth, k=top_k),
            ),
            EvaluationSummary(
                model_name="neural_collaborative_filtering",
                rmse=evaluate_rmse(test_df, ncf.predict_rating),
                precision_at_k=precision_at_k(collect_rankings(ncf), ground_truth, k=top_k),
                recall_at_k=recall_at_k(collect_rankings(ncf), ground_truth, k=top_k),
            ),
        ]

        self._save_artifacts(evaluations)
        self._refresh_models_from_disk()
        return evaluations

    def recommend(self, user_id: int, top_n: int = DEFAULT_TOP_K, model_name: str = "hybrid") -> List[Recommendation]:
        if model_name in {"hybrid", "user", "item"}:
            return self.cf.recommend(user_id, top_n=top_n, strategy=model_name)
        if model_name == "svd":
            return self.svd.recommend(user_id, top_n=top_n)
        if model_name == "ncf":
            return self.ncf.recommend(user_id, top_n=top_n)
        raise ValueError(f"Unsupported model '{model_name}'.")

    def get_metrics(self) -> List[Dict]:
        metrics_path = ARTIFACTS_DIR / "metrics.json"
        if not metrics_path.exists():
            return []
        return load_json(metrics_path).get("evaluations", [])

    def get_available_users(self) -> List[int]:
        return sorted(int(user_id) for user_id in self.ratings["userId"].dropna().unique().tolist())

    def get_supported_models(self) -> List[Dict[str, str]]:
        return [
            {"id": "hybrid", "label": "Hybrid vibe"},
            {"id": "user", "label": "User-based"},
            {"id": "item", "label": "Item-based"},
            {"id": "svd", "label": "SVD"},
            {"id": "ncf", "label": "Neural CF"},
        ]

    def _save_artifacts(self, evaluations: List[EvaluationSummary]) -> None:
        ensure_dir(ARTIFACTS_DIR)

        self.svd.fit()
        self.ncf.fit()

        np.save(ARTIFACTS_DIR / "svd_reconstructed.npy", self.svd.artifacts.reconstructed_matrix)
        torch.save(self.ncf.model.state_dict(), ARTIFACTS_DIR / "ncf_model.pt")
        save_json(
            ARTIFACTS_DIR / "metrics.json",
            {"evaluations": [asdict(summary) for summary in evaluations]},
        )

    def _refresh_models_from_disk(self) -> None:
        svd_path = ARTIFACTS_DIR / "svd_reconstructed.npy"
        ncf_path = ARTIFACTS_DIR / "ncf_model.pt"
        if svd_path.exists():
            reconstructed = np.load(svd_path)
            self.svd.fit()
            self.svd.artifacts.reconstructed_matrix = reconstructed
        if ncf_path.exists():
            state_dict = torch.load(ncf_path, map_location=self.ncf.device)
            self.ncf.model.load_state_dict(state_dict)
            self.ncf.is_trained = True


def bootstrap_service() -> RecommendationService:
    service = RecommendationService()
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    if not metrics_path.exists():
        service.train_and_evaluate()
    else:
        service._refresh_models_from_disk()
    return service
