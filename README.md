# Soil Testing Classification

End-to-end **Soil Testing Classification** project: classify soil type from soil and crop measurements using a Random Forest classifier plus a Streamlit dashboard.

## 1) Project Goal

- **Task:** Multi-class classification
- **Target column:** `Soil Type`
- **Classes:** `Black`, `Clayey`, `Loamy`, `Red`, `Sandy`
- **Input features used:**
  - `Temparature`
  - `Humidity`
  - `Moisture`
  - `Crop Type`
  - `Fertilizer Name`
  - `Nitrogen`
  - `Potassium`
  - `Phosphorous`
  - `ph`
  - `Soil_pH_Type`

## 2) Clean Project Structure

```text
Soil-Testing-Classification/
├─ app.py                      # Soil Testing Classification — Streamlit dashboard (single + batch prediction)
├─ train_rf.py                 # Train/evaluate model and save artifacts
├─ predict_rf.py               # Command-line prediction
├─ run_project.py              # One-command run flow
├─ feature_engineering.py      # Shared engineered features (used at train + inference)
├─ benchmark_models.py         # Optional multi-model benchmark
├─ requirements.txt            # Python dependencies
├─ README.md                   # Project documentation
├─ LICENSE
├─ Soil sample testing.csv     # Dataset
├─ soil_rf_model.joblib        # Trained model (generated)
├─ soil_label_encoder.joblib   # Label encoder (generated)
└─ outputs/
   └─ metrics.json             # Evaluation metrics (generated)
```

## 3) Requirements

- Python 3.10+ (works on Python 3.14 as well)
- `pip`

Install dependencies:

```bash
pip install -r requirements.txt
```

## 4) Complete Run Pipeline

### Option A: One command (recommended)

```bash
python run_project.py
```

This will:
1. Check required scripts and dataset
2. Train model if artifacts are missing
3. Run a sample prediction check
4. Launch Streamlit UI

### Option B: Manual steps

1. Train model

```bash
python train_rf.py
```

2. Run one prediction from terminal

```bash
python predict_rf.py --Temparature 30 --Humidity 55 --Moisture 45 --Crop_Type "Ground Nuts" --Nitrogen 8 --Potassium 4 --Phosphorous 18 --ph 7.2 --Fertilizer_Name "Ammonium Phosphate Complex" --Soil_pH_Type Alkaline
```

3. Launch **Soil Testing Classification** web app

```bash
streamlit run app.py
```

The browser tab title and header use the name **Soil Testing Classification**.

4. Benchmark multiple models (recommended for improvement)

```bash
python benchmark_models.py --sample-size 40000
```

This saves `outputs/model_benchmark.json` with Accuracy, Macro F1, and train time.

The `--sample-size` flag controls how many rows the benchmark uses. **Lower values are faster, the model ranking is unchanged** (the dataset's accuracy ceiling is feature-bound, see "Results & Limits"). Use **the full dataset** for the final report:

```bash
python benchmark_models.py --sample-size 0          # 0 = no subsampling, use all 100k rows
python benchmark_models.py --sample-size 100000     # equivalent
```

You can also chain everything (train if needed → CLI sanity check → full-dataset benchmark → Streamlit) with:

```bash
python run_project.py --benchmark
python run_project.py --benchmark --skip-app    # do not launch Streamlit at the end
```

## 5) What Each Script Does

### `train_rf.py`
- Loads `Soil sample testing.csv`
- Validates required columns
- Builds preprocessing pipeline:
  - Numeric: median imputation
  - Categorical: most-frequent imputation + one-hot encoding
- Splits data (stratified train/test)
- Adds engineered features (`ph_sq`, `ph_bin`) on top of all CSV predictors
- Trains `RandomForestClassifier` (500 trees, `max_depth=20`)
- **Two-phase fit (standard ML practice):**
  1. **Evaluation phase** — fit on the 80% train split, compute test accuracy, top-2, top-3, macro F1, full classification report and confusion matrix on the 20% hold-out, plus 5-fold stratified CV on the full dataset.
  2. **Deployment phase** — refit a fresh pipeline on **all 100,000 rows** so the saved model uses every available row.
- Reports **train accuracy, test accuracy, top-2, top-3 accuracy, macro F1, and 5-fold CV** so over/under-fitting is visible
- Saves:
  - `soil_rf_model.joblib` &nbsp;(fit on all 100k rows for deployment)
  - `soil_label_encoder.joblib`
  - `outputs/metrics.json` &mdash; includes `train_accuracy`, `top2_accuracy`, `top3_accuracy`, `cv5_accuracy_mean`, `cv5_accuracy_std`, `deployed_model_trained_on_rows`, and an `evaluation_protocol` note explaining what each number measures
  - `outputs/feature_importance.csv`

### `predict_rf.py`
- Loads saved model and label encoder
- Accepts input from CLI arguments
- Returns predicted soil type

### `app.py` (**Soil Testing Classification** dashboard)
- Single prediction form
- Confidence scores per class
- Batch CSV upload prediction
- Download prediction results CSV

## 6) Model Outputs

After training, you get:
- `soil_rf_model.joblib`: serialized trained Random Forest pipeline
- `soil_label_encoder.joblib`: maps encoded labels to original soil type names
- `outputs/metrics.json`: stores key evaluation metrics and confusion matrix

## 7) Results & Limits (Honest)

Latest run on the 20% stratified hold-out set:

| Metric | Value |
|--------|-------|
| Test accuracy | ~**0.396** |
| **Top-2 accuracy** | ~**0.635** |
| **Top-3 accuracy** | ~**0.846** |
| Macro F1 | ~**0.299** |
| 5-fold CV accuracy | ~**0.395** (+/- 0.001) |
| Train accuracy | ~**0.984** |

Diagnostics show why headline accuracy plateaus near **40%**:

- Only **`ph`** (and its discrete cousin **`Soil_pH_Type`**) carry real per-class signal. Per-class means and stds for `Temparature`, `Humidity`, `Moisture`, `Nitrogen`, `Potassium`, `Phosphorous`, `Crop Type`, and `Fertilizer Name` are nearly identical across the five soil types in this dataset.
- **Black** and **Red** have **identical** pH distributions (mean 6.50, std 0.58), so they are **statistically indistinguishable** with the available features. This is the structural reason their per-class recall stays low.
- The 5-fold CV result (~0.395 +/- 0.001) is extremely tight, confirming the ~40% test number is a **dataset/feature ceiling**, not training-run noise.

### Practical implications

- **Top-3** is high (~85%): the app's "Top 3 probable classes" view is genuinely useful even when top-1 is wrong.
- To push **top-1** higher, you need either richer/cleaner inputs, fewer (grouped) classes (e.g. merge Black + Red), or a different prediction target.

### Full-dataset benchmark (all 100,000 rows)

Stored in `outputs/model_benchmark.json` (run with `python benchmark_models.py --sample-size 0`). Each row records the **80/20 evaluation** (`accuracy`, `macro_f1`, `train_seconds`) **plus** the time to refit the same model on the full dataset (`full_data_refit_seconds`):

| Model | Test accuracy | Macro F1 | Eval train (s) | Full refit (s) |
|-------|---------------|----------|----------------|----------------|
| RandomForest | 0.3969 | 0.3025 | ~47 | ~45 |
| LinearSVC | 0.3946 | 0.2818 | ~7 | ~12 |
| ExtraTrees | 0.3882 | 0.3106 | ~65 | ~106 |
| LogisticRegression | 0.3759 | 0.3240 | ~123 | ~166 |

All four models cluster in the **~38–40%** band, which matches the dataset's information-theoretic ceiling. The ranking is the same on the 40k subsample (default) and on the full 100k dataset; only the wall-clock time changes.

Use `--skip-full-fit` to run only the evaluation phase (faster, no deployment-time numbers).

## 8) Important Notes

- Keep original column names exactly (especially `Temparature` spelling).
- `app.py` and `predict_rf.py` require model artifacts generated by training.
- For batch prediction, uploaded CSV must include all required input feature columns.

## 9) Troubleshooting

- **Error: model file not found**
  - Run: `python train_rf.py`

- **Module not found**
  - Run: `pip install -r requirements.txt`

- **Streamlit command not recognized**
  - Run app with: `python -m streamlit run app.py`