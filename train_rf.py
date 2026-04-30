import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


DATA_PATH = "Soil sample testing.csv"
TARGET_COL = "Soil Type"
DROP_COLS = ["Fertilizer Name", "Soil_pH_Type"]
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
        "ph",
        TARGET_COL,
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

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
        n_estimators=200,
        max_depth=16,
        random_state=42,
        n_jobs=1,
    )

    return Pipeline([("preprocessor", preprocessor), ("classifier", model)])


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    ensure_required_columns(df)

    X = df.drop(columns=[TARGET_COL] + [c for c in DROP_COLS if c in df.columns])
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

    pipeline = build_pipeline(X)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    cm_array = confusion_matrix(y_test, y_pred)
    cm = cm_array.tolist()
    report_dict = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
    )

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

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
        "macro_f1": round(float(macro_f1), 4),
        "classes": label_encoder.classes_.tolist(),
        "class_distribution": class_distribution,
        "confusion_matrix": cm,
        "classification_report": report_dict,
    }
    Path(METRICS_PATH).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Save global feature importance for explainability and reporting.
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out().tolist()
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": classifier.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved label encoder: {ENCODER_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved feature importance: {FEATURE_IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
