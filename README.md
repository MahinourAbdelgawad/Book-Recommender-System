# Book Recommender System Using Machine Learning

## Repository Structure

```
├── README.md
├── requirements.txt
├── preparation-and-design/
│   ├── README.md
│   ├── data_preparation/
│   ├── scripts/
│   ├── model_design/
│   ├── figures/
│   ├── logs/
│   ├── results/
│   ├── dataset/
│   └── dataset_original/
├── recommender_app/
│   ├── ml/
│   ├── model_service/
│   ├── web/
│   └── retraining/
├── tests/
└── instance/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Application

Open two terminals in `src/` and activate the virtual environment in both.

Start the model API:

```bash
python -m uvicorn recommender_app.model_service.api:app --host 127.0.0.1 --port 8000
```

In another terminal, start the Flask app:

```bash
python -m flask --app recommender_app.web run
```


## App Flow

**Create account → add or import books → rate five distinct books → Flask sends the reader profile and exclusions to the model API → hybrid PyTorch model scores catalog books → ranked recommendations are cached in SQLite and shown in the browser.**


The initial bundle is `instance/ml/v001/bundle/`; after promotion, the active bundle path is recorded in `instance/ml/active_bundle.txt`. The local library database is `instance/library.db`.

## Automatic retraining

The scheduler checks at a chosen interval. Once the rating threshold is met, it saves a frozen snapshot, trains a separate candidate, and compares it with the active model on validation data. A passing candidate is switched in only after the model API reloads and answers a smoke prediction. If reload or prediction fails, the prior active bundle is restored.


