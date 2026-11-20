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
