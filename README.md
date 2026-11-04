# Book Recommender System Using Machine Learning


## 1) Project Overview

The repository is organized as follows:

```text
src/
├── README.md
├── __init__.py
├── data_preparation/ # notebooks used in phase II
├── dataset/
├── dataset_original/
├── figures/ # for phase 2 report
├── logs/ # logs from each model run in pilot study
├── results/ # final results
│   ├── *.json # per-model evaluation results
│   ├── preds/ # saved model predictions for validation/test
│   └── summary_val.csv # leaderboard sorted by validation RMSE
├── scripts/ # scripts used for phase III pilot study
├── run_all_output.md
└── README.md
```

## 2) Phase II: Data Preparation

The notebooks in `data_preparation/` are the preprocessing and feature-engineering phase

- `1_check_datasets.ipynb`  
  Checks raw dataset files, schemas, missing values, and basic consistency.

- `2_extract_books.ipynb`  
  Cleans and standardizes the book table, merges metadata, and prepares book-side features.

- `3_extract_interactions.ipynb`  
  Loads user-book interactions, maps IDs, handles duplicates, and prepares the interaction table.

- `4_prepare_sample.ipynb`  
  Applies the sampling/filtering pipeline, including 20-core filtering, user sampling, and genre/author joins.

- `5_data_split.ipynb`  
  Splits data into train/validation/test, creates train-only user/book statistics, and prepares the final split structure.

- `6_preprocess_features.ipynb`  
  Builds final engineered features such as document embeddings, one-hot encodings, and standardized numeric features; also exports the final matrices.

### Output of this phase

The cleaned dataset is `dataset/cleaned_dataset.parquet` 



## 3) Phase III: Training / Model Evaluation

This is pilot study and training phase. It is controlled by the Python scripts in `scripts/`.


#### `scripts/prepare_data.py`
This is the script that prepared the model-ready training files for all later training runs. Found under `dataset/prepared/` 

#### `scripts/common.py`
This file contains the shared helpers used by all training scripts.


#### `scripts/models.py`
This file defines all the models used in the pilot study

#### `scripts/run_all.py`
This script runs the the full expereiment. It runs all the models in models.py in sequence and saves the resutls and logs.
The --fast option was used as a quick check to make sure the scripts were all fine before running the full training (sanity check)

#### `scripts/run_model.py`
This runs one model at a time. Used at a later stage to add models I initially forgot while the run_all script was running


#### `scripts/summarize.py`
This script reads the saved JSON results and prints a comparison table.
