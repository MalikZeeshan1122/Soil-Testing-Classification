import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC


DATA_PATH = "Soil sample testing.csv"
TARGET_COL = "Soil Type"
BENCHMARK_OUT = "outputs/model_benchmark.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark multiple models for soil type classification.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=40000,
        help="Optional random sample size for faster benchmarking. Use 0 for the full dataset.",
    )
    parser.add_argument(
        "--skip-full-fit",
        action="store_true",
        help="Skip the deployment-phase refit on the full dataset (eval-only, faster).",
    )
    return parser.parse_args()


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )


def main():
    args = parse_args()
    df = pd.read_csv(DATA_PATH)
    if args.sample_size and args.sample_size < len(df):
        df = df.sample(n=args.sample_size, random_state=42)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    preprocessor = make_preprocessor(X)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=20, random_state=42, n_jobs=1, class_weight="balanced_subsample"
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, max_depth=20, random_state=42, n_jobs=1, class_weight="balanced"
        ),
        "LogisticRegression": LogisticRegression(max_iter=3000, random_state=42),
        "LinearSVC": LinearSVC(random_state=42),
    }

    n_rows = int(len(df))
    results = []
    for name, model in models.items():
        # Phase 1: evaluation on stratified 80/20 split.
        start_eval = time.time()
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        eval_seconds = time.time() - start_eval

        row = {
            "model": name,
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_test, y_pred, average="macro")), 4),
            "train_seconds": round(eval_seconds, 2),
            "rows_used": n_rows,
        }

        # Phase 2: refit on the full benchmark sample for deployment-style timing.
        if not args.skip_full_fit:
            start_full = time.time()
            full_pipe = Pipeline([("preprocessor", preprocessor), ("classifier", model)])
            full_pipe.fit(X, y)
            row["full_data_refit_seconds"] = round(time.time() - start_full, 2)
            row["trained_on_full_dataset"] = True
        else:
            row["full_data_refit_seconds"] = None
            row["trained_on_full_dataset"] = False

        results.append(row)

    results = sorted(results, key=lambda x: (x["macro_f1"], x["accuracy"]), reverse=True)
    Path("outputs").mkdir(parents=True, exist_ok=True)
    Path(BENCHMARK_OUT).write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"Saved benchmark: {BENCHMARK_OUT}")


if __name__ == "__main__":
    main()
