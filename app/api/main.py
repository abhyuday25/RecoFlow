from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.services import RecommendationService, bootstrap_service


class RecommendationResponseItem(BaseModel):
    movie_id: int
    title: str
    genres: str
    score: float


class RecommendationResponse(BaseModel):
    user_id: int
    model: str
    recommendations: list[RecommendationResponseItem]


service: Optional[RecommendationService] = None
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    global service
    service = bootstrap_service()
    yield


app = FastAPI(title="Netflix-Style Recommender API", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict:
    if service is None:
        raise HTTPException(status_code=503, detail="Service is not initialized yet.")
    return {"evaluations": service.get_metrics()}


@app.get("/meta")
def meta() -> dict:
    if service is None:
        raise HTTPException(status_code=503, detail="Service is not initialized yet.")
    return {
        "app_name": "Netflix-Style Recommender API",
        "users": service.get_available_users(),
        "models": service.get_supported_models(),
        "metrics": service.get_metrics(),
    }


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend(user_id: int, model: str = Query(default="hybrid")) -> RecommendationResponse:
    if service is None:
        raise HTTPException(status_code=503, detail="Service is not initialized yet.")

    recommendations = service.recommend(user_id=user_id, top_n=10, model_name=model)
    if not recommendations:
        raise HTTPException(status_code=404, detail=f"No recommendations found for user {user_id}.")

    return RecommendationResponse(
        user_id=user_id,
        model=model,
        recommendations=[
            RecommendationResponseItem(
                movie_id=rec.movie_id,
                title=rec.title,
                genres=rec.genres,
                score=round(rec.score, 4),
            )
            for rec in recommendations
        ],
    )
