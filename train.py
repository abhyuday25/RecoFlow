from __future__ import annotations

from app.services import RecommendationService

def main() -> None:
    service = RecommendationService()
    evaluations = service.train_and_evaluate()
    print("Training complete.")
    for summary in evaluations:
        print(
            f"{summary.model_name}: RMSE={summary.rmse:.4f}, "
            f"Precision@10={summary.precision_at_k:.4f}, Recall@10={summary.recall_at_k:.4f}"
        )


if __name__ == "__main__":
    main()
