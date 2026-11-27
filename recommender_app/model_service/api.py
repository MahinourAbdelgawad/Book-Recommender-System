"""Local JSON API for the book rating and recommendation model."""

from contextlib import asynccontextmanager
import hmac
import os
from pathlib import Path
from threading import Lock
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field, StringConstraints
import torch

from recommender_app.ml.active_bundle import active_bundle_path
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


def authorize_reload(request: Request, x_api_key: Annotated[str | None, Header()] = None):
    expected = os.environ.get("MODEL_API_KEY")
    if not expected:
        loopback = {"127.0.0.1", "::1"}
        if (request.client is None or request.client.host not in loopback
                or request.url.hostname not in loopback | {"localhost"}
                or request.headers.get("origin")):
            raise HTTPException(status_code=403, detail="Unauthenticated model reload requires a local client")
        return
    if not hmac.compare_digest((x_api_key or "").encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Invalid model API key")


def get_predictor(request: Request) -> Predictor:
    return request.app.state.predictor


def create_app(bundle_dir: Path | None = None) -> FastAPI:
    def selected_bundle():
        return Path(bundle_dir).resolve() if bundle_dir else active_bundle_path(fallback=DEFAULT_BUNDLE)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        torch.set_num_threads(int(os.environ.get("MODEL_THREADS", "4")))
        app.state.predictor = Predictor(selected_bundle())
        yield

    app = FastAPI(title="Book Recommender Model API", lifespan=lifespan)
    reload_lock = Lock()

    @app.get("/health")
    def health(predictor: Annotated[Predictor, Depends(get_predictor)]):
        return {"status": "ok", "model_version": predictor.version,
                "catalog_size": len(predictor.books)}

    @app.post("/internal/reload-model", dependencies=[Depends(authorize_reload)])
    def reload_model(request: Request):
        # Construct fully before swapping so a broken candidate leaves the old predictor live.
        with reload_lock:
            try:
                directory = selected_bundle()
                predictor = Predictor(directory)
                if directory != selected_bundle():
                    raise ValueError("The active bundle changed during reload")
            except Exception as error:
                raise HTTPException(status_code=500, detail="Could not load selected model bundle") from error
            request.app.state.predictor = predictor
        return {"status": "ok", "model_version": predictor.version}

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
