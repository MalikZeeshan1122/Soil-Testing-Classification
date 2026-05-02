import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder

from feature_engineering import add_engineered_features


DATA_PATH = "Soil sample testing.csv"
TARGET_COL = "Soil Type"
MODEL_PATH = "soil_rf_model.joblib"
ENCODER_PATH = "soil_label_encoder.joblib"
METRICS_PATH = "outputs/metrics.json"
FEATURE_IMPORTANCE_PATH = "outputs/feature_importance.csv"


def ensure_required_columns(df: pd.DataFrame) -> None:
    required_cols = {
        "Temparature",
        "Humidity",
        "Moisture",
        "Crop Type",
        "Nitrogen",
        "Potassium",
        "Phosphorous",
        "Fertilizer Name",
        "ph",
        "Soil_pH_Type",
        TARGET_COL,
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def build_pipeline(X_engineered: pd.DataFrame) -> Pipeline:
    num_cols = X_engineered.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X_engineered.columns if c not in num_cols]

    preprocessor = ColumnTransformer(
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

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=1,
    )

    return Pipeline(
        [
            ("engineer", FunctionTransformer(add_engineered_features, validate=False)),
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    ensure_required_columns(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    eval_pipeline = build_pipeline(add_engineered_features(X.head(1)))
    eval_pipeline.fit(X_train, y_train)

    y_pred = eval_pipeline.predict(X_test)
    y_proba = eval_pipeline.predict_proba(X_test)
    train_acc = accuracy_score(y_train, eval_pipeline.predict(X_train))

    acc = accuracy_score(y_test, y_pred)
    top2_acc = top_k_accuracy_score(y_test, y_proba, k=2)
    top3_acc = top_k_accuracy_score(y_test, y_proba, k=3)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    cm_array = confusion_matrix(y_test, y_pred)
    cm = cm_array.tolist()
    report_dict = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
    )

    cv = cross_val_score(
        eval_pipeline,
        X,
        y_encoded,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=1,
    )

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test  accuracy: {acc:.4f}")
    print(f"Top-2 accuracy: {top2_acc:.4f}")
    print(f"Top-3 accuracy: {top3_acc:.4f}")
    print(f"Macro F1      : {macro_f1:.4f}")
    print(f"5-fold CV acc : {cv.mean():.4f} (+/- {cv.std():.4f})")
    print()
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Refit a fresh pipeline on the FULL dataset (all 100k rows) for deployment.
    # Reported metrics above remain the honest 80/20 hold-out evaluation; only
    # the saved artifact uses every available row, which is standard ML practice
    # once a model architecture has been validated.
    print("\nRefitting on the full dataset for deployment...")
    pipeline = build_pipeline(add_engineered_features(X.head(1)))
    pipeline.fit(X, y_encoded)

    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    Path("outputs").mkdir(parents=True, exist_ok=True)
    class_counts = y.value_counts().to_dict()
    class_distribution = {
        str(class_name): int(class_counts.get(class_name, 0))
        for class_name in label_encoder.classes_
    }
    metrics = {
        "accuracy": round(float(acc), 4),
        "train_accuracy": round(float(train_acc), 4),
        "top2_accuracy": round(float(top2_acc), 4),
        "top3_accuracy": round(float(top3_acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "cv5_accuracy_mean": round(float(cv.mean()), 4),
        "cv5_accuracy_std": round(float(cv.std()), 4),
        "deployed_model_trained_on_rows": int(len(X)),
        "evaluation_protocol": (
            "Metrics above (accuracy, top-k, macro_f1, train_accuracy, classification_report, "
            "confusion_matrix) come from a stratified 80/20 split. cv5_accuracy_* uses 5-fold "
            "stratified CV on the full dataset. The saved deployment model is then refit on all "
            "rows of Soil sample testing.csv."
        ),
        "classes": label_encoder.classes_.tolist(),
        "class_distribution": class_distribution,
        "confusion_matrix": cm,
        "classification_report": report_dict,
    }
    Path(METRICS_PATH).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out().tolist()
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": classifier.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved label encoder: {ENCODER_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved feature importance: {FEATURE_IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
