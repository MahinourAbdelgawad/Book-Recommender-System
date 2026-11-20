"""Local JSON API for the book rating and recommendation model."""

from contextlib import asynccontextmanager
import hmac
import os
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field, StringConstraints
import torch

from recommender_app.ml.predictor import Predictor

DEFAULT_BUNDLE = Path(__file__).resolve().parents[2] / "instance/ml/v001/bundle"
Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
Stars = Annotated[int, Field(strict=True, ge=1, le=5)]


class ReaderHistory(BaseModel):
    reader_id: Identifier | None = None
    ratings: dict[Identifier, Stars] = Field(default_factory=dict, max_length=50_000)


class PredictRequest(ReaderHistory):
    book_ids: list[Identifier] = Field(min_length=1, max_length=500)


class RecommendRequest(ReaderHistory):
    exclude_book_ids: list[Identifier] = Field(default_factory=list, max_length=50_000)
    top_n: int = Field(default=10, ge=1, le=100)


class BookScore(BaseModel):
    book_id: str
    predicted_rating: float


class Scores(BaseModel):
    model_version: str
    predictions: list[BookScore]


def check_key(x_api_key: Annotated[str | None, Header()] = None):
    expected = os.environ.get("MODEL_API_KEY")
    if expected and not hmac.compare_digest((x_api_key or "").encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Invalid model API key")


def get_predictor(request: Request) -> Predictor:
    return request.app.state.predictor


def create_app(bundle_dir: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        torch.set_num_threads(int(os.environ.get("MODEL_THREADS", "4")))
        directory = bundle_dir or Path(os.environ.get("MODEL_BUNDLE_DIR", DEFAULT_BUNDLE))
        app.state.predictor = Predictor(directory)
        yield

    app = FastAPI(title="Book Recommender Model API", lifespan=lifespan)

    @app.get("/health")
    def health(predictor: Annotated[Predictor, Depends(get_predictor)]):
        return {"status": "ok", "model_version": predictor.version,
                "catalog_size": len(predictor.books)}

    @app.post("/v1/predict", dependencies=[Depends(check_key)])
    def predict_books(body: PredictRequest, predictor: Annotated[Predictor, Depends(get_predictor)]) -> Scores:
        missing = list(dict.fromkeys(book for book in body.book_ids if book not in predictor.books))
        if missing:
            raise HTTPException(status_code=404, detail={"out_of_catalog": missing})
        ratings = predictor.score(body.book_ids, body.ratings, body.reader_id)
        return Scores(model_version=predictor.version, predictions=[
            BookScore(book_id=book, predicted_rating=rating) for book, rating in zip(body.book_ids, ratings)
        ])

    @app.post("/v1/recommend", dependencies=[Depends(check_key)])
    def recommend_books(body: RecommendRequest, predictor: Annotated[Predictor, Depends(get_predictor)]) -> Scores:
        ranked = predictor.recommend(body.ratings, body.reader_id, body.exclude_book_ids, body.top_n)
        return Scores(model_version=predictor.version, predictions=[
            BookScore(book_id=book, predicted_rating=rating) for book, rating in ranked
        ])

    return app


app = create_app()
