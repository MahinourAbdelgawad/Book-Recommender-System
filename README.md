# Book Recommendation System using Machine Learning

A book recommendation app that learns from reader ratings and suggests books from its catalog. Readers can import a Goodreads CSV or add books manually, and browse recommendations in a web app.

## Repository

```text
src/
├── preparation-and-design/   # Project phases, data preparation, and model experiments
├── recommender_app/
│   ├── ml/                   # Model, features, training, and evaluation
│   ├── model_service/        # FastAPI prediction and recommendation API
│   ├── web/                  # Flask app, templates, and static files
│   └── retraining/           # Rating snapshots, candidate training, scheduling, activation
├── instance/                 # Local database and model artifacts
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

For scheduled retraining, keep FastAPI on loopback and leave
`MODEL_BUNDLE_DIR` unset. `MODEL_API_KEY` is optional for local use.

```bash
python -m recommender_app.retraining.scheduler --interval-hours 1 --threshold 100
```

## Model

The recommender is a PyTorch hybrid neural network. It combines reader and book embeddings with engineered reader and book features to predict a rating. The selected model has two hidden layers, with 128 and 64 units.

## Application

The app uses Flask for the web interface, FastAPI for model predictions, and SQLite for accounts, libraries, and recommendation history. It supports Goodreads CSV import, manual book entry, recommendations, and reader feedback through library ratings and book actions.

## Retraining

The scheduler periodically checks for enough new eligible ratings. When the threshold is met, it saves a rating snapshot, trains a separate candidate, and compares it with the active model on validation data. A candidate that passes is activated through the model API. If reload or a prediction check fails, the scheduler restores the previous model. Choose the threshold when starting the scheduler.
