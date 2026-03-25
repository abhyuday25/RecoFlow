from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app.data.dataset import EncodedData, build_seen_items_map
from app.models.collaborative_filtering import Recommendation


class RatingsDataset(Dataset):
    def __init__(self, user_indices: np.ndarray, item_indices: np.ndarray, ratings: np.ndarray) -> None:
        self.user_indices = torch.tensor(user_indices, dtype=torch.long)
        self.item_indices = torch.tensor(item_indices, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int):
        return self.user_indices[idx], self.item_indices[idx], self.ratings[idx]


class NeuralCollaborativeFilteringModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 16) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        user_embedding = self.user_embedding(user_indices)
        item_embedding = self.item_embedding(item_indices)
        features = torch.cat([user_embedding, item_embedding], dim=1)
        return self.mlp(features).squeeze(-1)


@dataclass
class NeuralTrainingArtifacts:
    model_state_dict: dict
    final_loss: float


class NeuralCFRecommender:
    def __init__(self, encoded_data: EncodedData, embedding_dim: int = 16, device: str | None = None) -> None:
        self.data = encoded_data
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NeuralCollaborativeFilteringModel(
            num_users=len(encoded_data.user_to_index),
            num_items=len(encoded_data.item_to_index),
            embedding_dim=embedding_dim,
        ).to(self.device)
        self.seen_items = build_seen_items_map(encoded_data.ratings)
        self.movies_indexed = encoded_data.movies.set_index("movieId")
        self.is_trained = False

    def fit(self, epochs: int = 25, learning_rate: float = 1e-3, batch_size: int = 32) -> NeuralTrainingArtifacts:
        ratings = self.data.ratings.copy()
        user_indices = ratings["userId"].map(self.data.user_to_index).to_numpy()
        item_indices = ratings["movieId"].map(self.data.item_to_index).to_numpy()
        targets = ratings["normalized_rating"].to_numpy()

        dataset = RatingsDataset(user_indices, item_indices, targets)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()
        final_loss = 0.0
        for _ in range(epochs):
            total_loss = 0.0
            for users, items, batch_ratings in loader:
                users = users.to(self.device)
                items = items.to(self.device)
                batch_ratings = batch_ratings.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(users, items)
                loss = criterion(predictions, batch_ratings)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
            final_loss = total_loss / max(len(loader), 1)

        self.is_trained = True
        return NeuralTrainingArtifacts(model_state_dict=self.model.state_dict(), final_loss=final_loss)

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        if not self.is_trained:
            self.fit()

        if user_id not in self.data.user_to_index or movie_id not in self.data.item_to_index:
            return float(self.data.ratings["rating"].mean())

        self.model.eval()
        with torch.no_grad():
            user_idx = torch.tensor([self.data.user_to_index[user_id]], dtype=torch.long, device=self.device)
            item_idx = torch.tensor([self.data.item_to_index[movie_id]], dtype=torch.long, device=self.device)
            normalized_score = float(self.model(user_idx, item_idx).cpu().item())

        min_rating = self.data.ratings["rating"].min()
        max_rating = self.data.ratings["rating"].max()
        rating = normalized_score * (max_rating - min_rating) + min_rating
        return float(np.clip(rating, min_rating, max_rating))

    def recommend(self, user_id: int, top_n: int = 10) -> List[Recommendation]:
        if not self.is_trained:
            self.fit()

        if user_id not in self.data.user_to_index:
            return []

        seen = self.seen_items.get(user_id, set())
        candidates = [movie_id for movie_id in self.data.item_to_index if movie_id not in seen]
        if not candidates:
            return []

        user_idx = self.data.user_to_index[user_id]
        user_tensor = torch.tensor([user_idx] * len(candidates), dtype=torch.long, device=self.device)
        item_tensor = torch.tensor(
            [self.data.item_to_index[movie_id] for movie_id in candidates], dtype=torch.long, device=self.device
        )

        self.model.eval()
        with torch.no_grad():
            scores = self.model(user_tensor, item_tensor).cpu().numpy()

        ranked_pairs = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)[:top_n]
        recommendations = []
        for movie_id, score in ranked_pairs:
            movie = self.movies_indexed.loc[movie_id]
            recommendations.append(
                Recommendation(
                    movie_id=movie_id,
                    score=float(score),
                    title=str(movie["title"]),
                    genres=str(movie["genres"]),
                )
            )
        return recommendations
