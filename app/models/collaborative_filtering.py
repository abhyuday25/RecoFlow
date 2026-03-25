from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from app.data.dataset import EncodedData, build_seen_items_map


@dataclass
class Recommendation:
    movie_id: int
    score: float
    title: str
    genres: str


class CollaborativeFilteringRecommender:
    def __init__(self, encoded_data: EncodedData) -> None:
        self.data = encoded_data
        self.user_similarity = cosine_similarity(encoded_data.normalized_matrix)
        self.item_similarity = cosine_similarity(encoded_data.normalized_matrix.T)
        self.seen_items = build_seen_items_map(encoded_data.ratings)
        self.movies_indexed = encoded_data.movies.set_index("movieId")

    def _recommend_user_based(self, user_id: int, top_n: int = 10) -> List[Recommendation]:
        if user_id not in self.data.user_to_index:
            return []

        user_idx = self.data.user_to_index[user_id]
        user_scores = self.user_similarity[user_idx]
        weighted_scores = user_scores @ self.data.normalized_matrix
        similarity_sum = np.abs(user_scores).sum() + 1e-8
        predicted_scores = weighted_scores / similarity_sum
        return self._format_recommendations(user_id, predicted_scores, top_n)

    def _recommend_item_based(self, user_id: int, top_n: int = 10) -> List[Recommendation]:
        if user_id not in self.data.user_to_index:
            return []

        user_idx = self.data.user_to_index[user_id]
        user_vector = self.data.normalized_matrix[user_idx]
        weighted_scores = user_vector @ self.item_similarity
        similarity_sum = np.abs(self.item_similarity).sum(axis=0) + 1e-8
        predicted_scores = weighted_scores / similarity_sum
        return self._format_recommendations(user_id, predicted_scores, top_n)

    def recommend(self, user_id: int, top_n: int = 10, strategy: str = "hybrid") -> List[Recommendation]:
        if strategy == "user":
            return self._recommend_user_based(user_id, top_n)
        if strategy == "item":
            return self._recommend_item_based(user_id, top_n)

        user_recs = self._recommend_user_based(user_id, top_n * 2)
        item_recs = self._recommend_item_based(user_id, top_n * 2)

        merged_scores: Dict[int, float] = {}
        for rec in user_recs:
            merged_scores[rec.movie_id] = merged_scores.get(rec.movie_id, 0.0) + rec.score
        for rec in item_recs:
            merged_scores[rec.movie_id] = merged_scores.get(rec.movie_id, 0.0) + rec.score

        ranked = sorted(merged_scores.items(), key=lambda pair: pair[1], reverse=True)[:top_n]
        recommendations = []
        for movie_id, score in ranked:
            movie = self.movies_indexed.loc[movie_id]
            recommendations.append(
                Recommendation(
                    movie_id=movie_id,
                    score=float(score / 2.0),
                    title=str(movie["title"]),
                    genres=str(movie["genres"]),
                )
            )
        return recommendations

    def predict_rating(self, user_id: int, movie_id: int, strategy: str = "hybrid") -> float:
        if user_id not in self.data.user_to_index or movie_id not in self.data.item_to_index:
            return float(self.data.ratings["rating"].mean())

        user_idx = self.data.user_to_index[user_id]
        item_idx = self.data.item_to_index[movie_id]

        user_based = (
            (self.user_similarity[user_idx] @ self.data.raw_matrix[:, item_idx])
            / (np.abs(self.user_similarity[user_idx]).sum() + 1e-8)
        )
        item_based = (
            (self.data.raw_matrix[user_idx] @ self.item_similarity[:, item_idx])
            / (np.abs(self.item_similarity[:, item_idx]).sum() + 1e-8)
        )

        if strategy == "user":
            return float(user_based)
        if strategy == "item":
            return float(item_based)
        return float((user_based + item_based) / 2.0)

    def _format_recommendations(self, user_id: int, predicted_scores: np.ndarray, top_n: int) -> List[Recommendation]:
        seen = self.seen_items.get(user_id, set())
        ranked_indices = np.argsort(predicted_scores)[::-1]
        recommendations: List[Recommendation] = []

        for item_idx in ranked_indices:
            movie_id = self.data.index_to_item[item_idx]
            if movie_id in seen:
                continue
            movie = self.movies_indexed.loc[movie_id]
            recommendations.append(
                Recommendation(
                    movie_id=movie_id,
                    score=float(predicted_scores[item_idx]),
                    title=str(movie["title"]),
                    genres=str(movie["genres"]),
                )
            )
            if len(recommendations) >= top_n:
                break

        return recommendations
