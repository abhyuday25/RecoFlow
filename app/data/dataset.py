from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class EncodedData:
    ratings: pd.DataFrame
    movies: pd.DataFrame
    user_to_index: Dict[int, int]
    index_to_user: Dict[int, int]
    item_to_index: Dict[int, int]
    index_to_item: Dict[int, int]
    normalized_matrix: np.ndarray
    raw_matrix: np.ndarray


def load_movielens_data(ratings_path: str, movies_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    ratings = ratings.dropna(subset=["userId", "movieId", "rating"]).copy()
    movies = movies.dropna(subset=["movieId", "title"]).copy()

    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)
    movies["movieId"] = movies["movieId"].astype(int)
    movies["genres"] = movies["genres"].fillna("Unknown")

    return ratings, movies


def normalize_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    normalized = ratings.copy()
    min_rating = normalized["rating"].min()
    max_rating = normalized["rating"].max()
    if max_rating == min_rating:
        normalized["normalized_rating"] = 0.0
    else:
        normalized["normalized_rating"] = (normalized["rating"] - min_rating) / (max_rating - min_rating)
    return normalized


def encode_dataset(ratings: pd.DataFrame, movies: pd.DataFrame) -> EncodedData:
    ratings = normalize_ratings(ratings)
    unique_users = sorted(ratings["userId"].unique())
    unique_items = sorted(movies["movieId"].unique())

    user_to_index = {user_id: idx for idx, user_id in enumerate(unique_users)}
    item_to_index = {item_id: idx for idx, item_id in enumerate(unique_items)}
    index_to_user = {idx: user_id for user_id, idx in user_to_index.items()}
    index_to_item = {idx: item_id for item_id, idx in item_to_index.items()}

    raw_matrix = np.zeros((len(unique_users), len(unique_items)), dtype=np.float32)
    normalized_matrix = np.zeros_like(raw_matrix)

    for row in ratings.itertuples(index=False):
        user_idx = user_to_index.get(row.userId)
        item_idx = item_to_index.get(row.movieId)
        if user_idx is None or item_idx is None:
            continue
        raw_matrix[user_idx, item_idx] = row.rating
        normalized_matrix[user_idx, item_idx] = row.normalized_rating

    return EncodedData(
        ratings=ratings,
        movies=movies,
        user_to_index=user_to_index,
        index_to_user=index_to_user,
        item_to_index=item_to_index,
        index_to_item=index_to_item,
        normalized_matrix=normalized_matrix,
        raw_matrix=raw_matrix,
    )


def build_seen_items_map(ratings: pd.DataFrame) -> Dict[int, set]:
    grouped = ratings.groupby("userId")["movieId"].apply(set)
    return grouped.to_dict()


def train_test_split_ratings(
    ratings: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(ratings, test_size=test_size, random_state=random_state)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_ground_truth(test_df: pd.DataFrame, threshold: float = 4.0) -> Dict[int, List[int]]:
    filtered = test_df[test_df["rating"] >= threshold]
    grouped = filtered.groupby("userId")["movieId"].apply(list)
    return grouped.to_dict()
