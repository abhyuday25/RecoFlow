from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.decomposition import TruncatedSVD

from app.data.dataset import EncodedData, build_seen_items_map
from app.models.collaborative_filtering import Recommendation


@dataclass
class MatrixFactorizationArtifacts:
    reconstructed_matrix: np.ndarray
    explained_variance: float


class SVDRecommender:
    def __init__(self, encoded_data: EncodedData, n_factors: int = 8, random_state: int = 42) -> None:
        self.data = encoded_data
        self.n_factors = max(2, min(n_factors, min(encoded_data.normalized_matrix.shape) - 1))
        self.random_state = random_state
        self.model = TruncatedSVD(n_components=self.n_factors, random_state=random_state)
        self.artifacts: MatrixFactorizationArtifacts | None = None
        self.seen_items = build_seen_items_map(encoded_data.ratings)
        self.movies_indexed = encoded_data.movies.set_index("movieId")

    def fit(self) -> MatrixFactorizationArtifacts:
        transformed = self.model.fit_transform(self.data.normalized_matrix)
        reconstructed = transformed @ self.model.components_
        self.artifacts = MatrixFactorizationArtifacts(
            reconstructed_matrix=reconstructed,
            explained_variance=float(self.model.explained_variance_ratio_.sum()),
        )
        return self.artifacts

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        if self.artifacts is None:
            self.fit()

        if user_id not in self.data.user_to_index or movie_id not in self.data.item_to_index:
            return float(self.data.ratings["rating"].mean())

        user_idx = self.data.user_to_index[user_id]
        item_idx = self.data.item_to_index[movie_id]
        normalized_score = self.artifacts.reconstructed_matrix[user_idx, item_idx]

        min_rating = self.data.ratings["rating"].min()
        max_rating = self.data.ratings["rating"].max()
        rating = normalized_score * (max_rating - min_rating) + min_rating
        return float(np.clip(rating, min_rating, max_rating))

    def recommend(self, user_id: int, top_n: int = 10) -> List[Recommendation]:
        if self.artifacts is None:
            self.fit()

        if user_id not in self.data.user_to_index:
            return []

        user_idx = self.data.user_to_index[user_id]
        scores = self.artifacts.reconstructed_matrix[user_idx]
        seen = self.seen_items.get(user_id, set())
        ranked_indices = np.argsort(scores)[::-1]

        recommendations: List[Recommendation] = []
        for item_idx in ranked_indices:
            movie_id = self.data.index_to_item[item_idx]
            if movie_id in seen:
                continue
            movie = self.movies_indexed.loc[movie_id]
            recommendations.append(
                Recommendation(
                    movie_id=movie_id,
                    score=float(scores[item_idx]),
                    title=str(movie["title"]),
                    genres=str(movie["genres"]),
                )
            )
            if len(recommendations) >= top_n:
                break
        return recommendations
