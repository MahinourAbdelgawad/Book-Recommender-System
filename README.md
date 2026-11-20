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

Run from `src/` with the project's virtual environment active. The ML code uses
NumPy, pandas, PyArrow, PyTorch and Matplotlib. The default input is the local
Phase II engineered ratings parquet under `preparation-and-design/dataset/`.

```bash
python -m recommender_app.ml.parity
python -u -m recommender_app.ml.experiment sweep --version v001
```

The parity check uses the trusted local Phase IV notebook and checkpoint. The
sweep trains nine models and can take time. It writes each completed fit's
validation metrics, learning curves and checkpoint under `instance/ml/v001/`.
Use a fresh version name for a new sweep. The selected bundle and PNG are written
when all nine fits finish.

The sweep compares one (64), two (128, 64), and three (256, 128, 64) hidden layers
over seeds 0, 1, and 2. All use 64-dimensional embeddings, 0.10 dropout, batch
normalization and the Phase IV optimizer settings. Early stopping monitors warm
validation RMSE. Curves show clipped RMSE in evaluation mode for both splits,
at the same weights each epoch. The gap is validation minus training RMSE.

Selection uses the mean of warm validation RMSE and five-rating cold validation
RMSE. Relative to the three-layer control, a candidate can lose at most 0.002
warm RMSE, 0.005 cold RMSE, and 0.01 cold HR@10 (100 sampled unrated books).
Within 0.001 of the best eligible objective, prefer fewer parameters and then a
smaller absolute gap. Export the seed closest to the selected depth's mean
objective. These are selection tolerances, without any statistical significance claim.

User cohorts and warm features follow Phase IV. Cold validation excludes rows
from the original test split and still reserves the first 20 eligible interactions
for possible context, revealing five. This makes the new cold validation numbers
a separate evaluation from the notebook's headline cold result. Static book
inputs come from the engineered parquet. This phase rebuilds user and book rating
aggregates and their scalers from warm training rows. For Phase IV parity,
training means exclude the row’s rating, while training reader standard deviation
uses the full warm training history. That difference matters when interpreting
the train–validation gap. New-book featurization
from raw metadata belongs to the later catalog expansion work.

After inspecting the validation curves and freezing the selected bundle, run:

```bash
python -m recommender_app.ml.experiment evaluate-test --version v001
```

This evaluates the frozen weights on warm test rows and the separate held-out cold
test cohort, writing `test_metrics.json`. Repeating the command for the same
version displays the saved result.
