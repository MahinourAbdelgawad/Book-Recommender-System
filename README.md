# Book Recommendation System using Machine Learning

A book recommendation app that learns from reader ratings and suggests books from its catalog. Readers can import a Goodreads CSV or add books manually, and browse recommendations in a web app.

## Repository

```text
src/
├── instance/                 # Local database and model artifacts
├── preparation-and-design/   # Project phases, data preparation, and model experiments
├── recommender_app/
│   ├── ml/                   # Model, features, training, and evaluation
│   ├── model_service/        # FastAPI prediction and recommendation API
│   ├── web/                  # Flask app, templates, and static files
│   └── retraining/           # Rating snapshots, candidate training, scheduling, activation
├── tests/
├── requirements.txt
└── README.md
```

## Setup and run

From this directory, create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the model API, Flask app, and training scheduler in separate terminals:

```bash
python -m uvicorn recommender_app.model_service.api:app --host 127.0.0.1 --port 8000
```

```bash
python -m flask --app recommender_app.web run
```

```bash
python -m recommender_app.retraining.scheduler --interval-hours 1 --threshold 100
```


## Model

The recommender is a PyTorch hybrid neural network. It combines reader and book embeddings with engineered reader and book features to predict a rating. The selected model has two hidden layers, with 128 and 64 units.

## Application

The app uses Flask for the web interface, FastAPI for model predictions, and SQLite for accounts, libraries, and recommendation history. It supports Goodreads CSV import, manual book entry, recommendations, and reader feedback through library ratings and book actions.

## Model API examples

### POST /v1/predict

Predicts ratings for the requested books.

Request:

```json
{
  "reader_id": "app:12",
  "ratings": {"101": 5, "102": 4, "103": 3, "104": 5, "105": 4},
  "book_ids": ["201", "202"]
}
```

Response (200 OK):

```json
{
  "model_version": "v001",
  "predictions": [
    {"book_id": "201", "predicted_rating": 4.36},
    {"book_id": "202", "predicted_rating": 3.89}
  ]
}
```

If a requested book is outside the active catalog, the response is 404:

```json
{"detail": {"out_of_catalog": ["999999999"]}}
```

### POST /v1/recommend

Returns up to `top_n` books, ranked by predicted rating after applying exclusions.

Request:

```json
{
  "reader_id": "app:12",
  "ratings": {"101": 5, "102": 4, "103": 3, "104": 5, "105": 4},
  "exclude_book_ids": ["205", "206"],
  "top_n": 2
}
```

Response (200 OK):

```json
{
  "model_version": "v001",
  "predictions": [
    {"book_id": "301", "predicted_rating": 4.71},
    {"book_id": "302", "predicted_rating": 4.58}
  ]
}
```

### GET /health

Send the request without a body. Response (200 OK):

```json
{"status": "ok", "model_version": "v001", "catalog_size": 22000}
```

### POST /internal/reload-model

The scheduler updates the active bundle pointer, then calls this endpoint
without a request body. FastAPI loads the selected bundle and returns its
version. The scheduler checks health and scoring separately.

Response (200 OK):

```json
{"status": "ok", "model_version": "v002"}
```

## Retraining

The scheduler periodically checks for enough new eligible ratings. When the threshold is met, it saves a rating snapshot, trains a separate candidate, and compares it with the active model on validation data. A candidate that passes is activated through the model API. If reload or a prediction check fails, the scheduler restores the previous model. Choose the threshold when starting the scheduler.
