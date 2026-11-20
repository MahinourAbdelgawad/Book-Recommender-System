# Book Recommender System Using Machine Learning

### Repository Structure:
```
├── README.md 
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

### Final model depth experiment

```bash
# Check that the extracted features and predictions match the ones in Phase IV
python -m recommender_app.ml.parity

# Compare one, two, and three hidden layers on validation and save the winner
python -u -m recommender_app.ml.experiment sweep --version v001

# Evaluate the frozen winner on the test data
python -m recommender_app.ml.experiment evaluate-test --version v001
```

**Selected model:** Two hidden layers (128 and 64 units), 64-dimensional embeddings,
and 0.10 dropout. The exported checkpoint uses seed 2 at epoch 2.

The bundle, learning curves, and metrics are in `instance/ml/v001/`

### Model API

Run from `src/` with the project virtual environment active.

```bash
# Install the API and test dependencies.
python -m pip install -r requirements-model-service.txt
# One-time setup for the existing v001 bundle.
python -m recommender_app.model_service.index_readers
# Serve v001 locally.
python -m uvicorn recommender_app.model_service.api:app --host 127.0.0.1 --port 8000
# Check direct-vs-API parity without a running server.
python -m unittest discover -s tests -p test_model_service.py
```

Open `http://127.0.0.1:8000/docs` to try the API.

- `GET /health` returns the model version and catalog size.
- `POST /v1/predict` accepts `book_ids` and returns a score per catalog book, in request order.
- `POST /v1/recommend` accepts `top_n` and `exclude_book_ids`, and ranks eligible books by score. Rated books are excluded automatically. Pass read-but-unrated books in the exclusion list.
- Both POST endpoints accept `ratings` as a map from string book IDs to stars, e.g. `{"123": 4}`, and an optional `reader_id`. Ratings of out-of-catalog books contribute to the reader profile. Supply the reader ID only when it belongs to the bundle's identity namespace. Other readers use the unknown embedding.

Set `MODEL_BUNDLE_DIR` to load another exported bundle or `MODEL_API_KEY` to require an `X-API-Key` on prediction and recommendation requests. New bundles include the trained-reader index automatically.
