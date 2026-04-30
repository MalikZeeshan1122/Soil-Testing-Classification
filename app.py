import io
import pandas as pd
import streamlit as st
import joblib


MODEL_PATH = "soil_rf_model.joblib"
ENCODER_PATH = "soil_label_encoder.joblib"
REQUIRED_COLUMNS = [
    "Temparature",
    "Humidity",
    "Moisture",
    "Crop Type",
    "Nitrogen",
    "Potassium",
    "Phosphorous",
    "ph",
]


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder


def single_prediction_ui(model, encoder):
    st.subheader("Single Prediction")
    st.caption("Enter soil measurements and crop information.")

    col1, col2 = st.columns(2)
    with col1:
        temparature = st.number_input("Temparature", min_value=0.0, max_value=60.0, value=30.0)
        humidity = st.number_input("Humidity", min_value=0.0, max_value=100.0, value=55.0)
        moisture = st.number_input("Moisture", min_value=0.0, max_value=100.0, value=45.0)
        crop_type = st.text_input("Crop Type", value="Ground Nuts")
    with col2:
        nitrogen = st.number_input("Nitrogen", min_value=0.0, max_value=200.0, value=8.0)
        potassium = st.number_input("Potassium", min_value=0.0, max_value=200.0, value=4.0)
        phosphorous = st.number_input("Phosphorous", min_value=0.0, max_value=200.0, value=18.0)
        ph = st.number_input("ph", min_value=0.0, max_value=14.0, value=7.2)

    if st.button("Predict Soil Type", type="primary"):
        input_df = pd.DataFrame(
            [
                {
                    "Temparature": temparature,
                    "Humidity": humidity,
                    "Moisture": moisture,
                    "Crop Type": crop_type,
                    "Nitrogen": nitrogen,
                    "Potassium": potassium,
                    "Phosphorous": phosphorous,
                    "ph": ph,
                }
            ]
        )

        pred_idx = model.predict(input_df)[0]
        pred_label = encoder.inverse_transform([pred_idx])[0]
        st.success(f"Predicted Soil Type: {pred_label}")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            confidence_df = pd.DataFrame(
                {"Soil Type": encoder.classes_, "Confidence": probs}
            ).sort_values("Confidence", ascending=False)
            st.markdown("#### Class Confidence")
            st.dataframe(confidence_df, use_container_width=True, hide_index=True)
            st.bar_chart(confidence_df.set_index("Soil Type")["Confidence"])


def batch_prediction_ui(model, encoder):
    st.subheader("Batch Prediction")
    st.caption("Upload a CSV with required columns to predict multiple rows.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    st.write("Required columns:", ", ".join(REQUIRED_COLUMNS))

    if not uploaded_file:
        return

    try:
        batch_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read CSV file: {exc}")
        return

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in batch_df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return

    inference_df = batch_df[REQUIRED_COLUMNS].copy()
    pred_idx = model.predict(inference_df)
    pred_labels = encoder.inverse_transform(pred_idx)

    result_df = batch_df.copy()
    result_df["Predicted Soil Type"] = pred_labels

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(inference_df)
        result_df["Prediction Confidence"] = probs.max(axis=1)

    st.markdown("#### Prediction Output")
    st.dataframe(result_df.head(50), use_container_width=True)
    st.caption("Showing first 50 rows in app preview.")

    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Full Results CSV",
        data=csv_buffer.getvalue(),
        file_name="soil_type_predictions.csv",
        mime="text/csv",
    )


def main():
    st.set_page_config(page_title="Soil Type Classifier", page_icon="🌱", layout="wide")
    st.title("Soil Type Classification App")
    st.write("Random Forest model for soil type prediction from soil measurements.")

    try:
        model, encoder = load_artifacts()
    except Exception as exc:
        st.error(
            "Model files not found or failed to load. "
            "Train and save these files first: "
            "`soil_rf_model.joblib`, `soil_label_encoder.joblib`."
        )
        st.exception(exc)
        return

    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
    with tab1:
        single_prediction_ui(model, encoder)
    with tab2:
        batch_prediction_ui(model, encoder)

    st.markdown("---")
    st.caption("Tip: Keep column names exactly as used in training, including 'Temparature'.")


if __name__ == "__main__":
    main()
