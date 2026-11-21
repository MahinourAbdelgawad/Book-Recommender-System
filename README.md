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


The selected bundle is stored in `instance/ml/v001/bundle/`. The local library database is `instance/library.db`.
