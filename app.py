import io
import json
import pandas as pd
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


MODEL_PATH = "soil_rf_model.joblib"
ENCODER_PATH = "soil_label_encoder.joblib"
BENCHMARK_PATH = "outputs/model_benchmark.json"
DATA_PATH = "Soil sample testing.csv"
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


def apply_custom_style():
    st.markdown(
        """
        <style>
            .main-title {
                font-size: 2.1rem;
                font-weight: 700;
                margin-bottom: 0.2rem;
            }
            .sub-title {
                color: #7a7a7a;
                margin-bottom: 1.2rem;
            }
            .metric-card {
                background: linear-gradient(135deg, #f6fff8 0%, #eef7ff 100%);
                border: 1px solid #dbe8dd;
                border-radius: 14px;
                padding: 0.8rem 1rem;
                margin-bottom: 0.6rem;
            }
            .metric-label {
                font-size: 0.85rem;
                color: #4d4d4d;
                margin-bottom: 0.2rem;
            }
            .metric-value {
                font-size: 1.3rem;
                font-weight: 700;
                color: #173f2e;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_plotly(fig):
    fig.update_layout(
        template="plotly_white",
        legend_title_text="",
        margin=dict(l=20, r=20, t=55, b=20),
        title=dict(font=dict(size=18)),
    )
    return fig


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder


@st.cache_data
def load_benchmark_results():
    try:
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if isinstance(rows, list) and rows:
            return pd.DataFrame(rows)
    except Exception:
        return None
    return None


@st.cache_data
def load_reference_data():
    try:
        df = pd.read_csv(DATA_PATH)
        needed = REQUIRED_COLUMNS + ["Soil Type"]
        if all(col in df.columns for col in needed):
            return df[needed].copy()
    except Exception:
        return None
    return None


def explain_prediction_block(input_df, confidence_df):
    st.markdown("#### Why This Prediction?")
    top3 = confidence_df.head(3).copy()
    top3["Confidence"] = top3["Confidence"].map(lambda x: f"{x:.2%}")
    st.write("Top 3 probable classes")
    st.dataframe(top3, use_container_width=True, hide_index=True)

    top_val = float(confidence_df.iloc[0]["Confidence"].replace("%", "")) if isinstance(confidence_df.iloc[0]["Confidence"], str) else float(confidence_df.iloc[0]["Confidence"])
    second_val = float(confidence_df.iloc[1]["Confidence"].replace("%", "")) if isinstance(confidence_df.iloc[1]["Confidence"], str) else float(confidence_df.iloc[1]["Confidence"])
    gap = abs(top_val - second_val)
    if gap < 0.08:
        st.info(
            "Top classes are close to each other. This input appears near class overlap boundaries in the training data."
        )
    else:
        st.info(
            "Top class is clearly ahead of the second class, so the model sees a stronger class pattern."
        )

    ref_df = load_reference_data()
    if ref_df is None:
        st.caption("Reference dataset unavailable for similarity examples.")
        return

    num_cols = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous", "ph"]
    crop_val = input_df.iloc[0]["Crop Type"]
    candidate_df = ref_df[ref_df["Crop Type"] == crop_val].copy()
    if candidate_df.empty:
        candidate_df = ref_df.copy()

    means = candidate_df[num_cols].mean()
    stds = candidate_df[num_cols].std().replace(0, 1).fillna(1)
    input_vec = ((input_df.iloc[0][num_cols] - means) / stds).astype(float).to_numpy()
    cand_scaled = ((candidate_df[num_cols] - means) / stds).astype(float)
    dists = np.sqrt(((cand_scaled - input_vec) ** 2).sum(axis=1))
    nearest = candidate_df.assign(distance=dists).sort_values("distance").head(5)
    st.write("Most similar training examples (nearest profiles)")
    st.dataframe(
        nearest[
            [
                "Soil Type",
                "Temparature",
                "Humidity",
                "Moisture",
                "Crop Type",
                "Nitrogen",
                "Potassium",
                "Phosphorous",
                "ph",
                "distance",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


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
            top_confidence = float(probs.max())
            confidence_df = pd.DataFrame(
                {"Soil Type": encoder.classes_, "Confidence": probs}
            ).sort_values("Confidence", ascending=False)
            top_3 = confidence_df.head(3).reset_index(drop=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-label'>Top Prediction</div><div class='metric-value'>{pred_label}</div></div>",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-label'>Top Confidence</div><div class='metric-value'>{top_confidence:.2%}</div></div>",
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-label'>2nd Best Class</div><div class='metric-value'>{top_3.loc[1, 'Soil Type'] if len(top_3) > 1 else '-'}</div></div>",
                    unsafe_allow_html=True,
                )

            if top_confidence < 0.50:
                st.warning(
                    "Low-confidence prediction. Treat this result as uncertain and verify with more data."
                )

            st.markdown("#### Class Confidence")
            st.dataframe(confidence_df, use_container_width=True, hide_index=True)

            col_chart_1, col_chart_2 = st.columns(2)
            with col_chart_1:
                bar_fig = px.bar(
                    confidence_df,
                    x="Soil Type",
                    y="Confidence",
                    title="Confidence by Soil Type (Bar)",
                    text=confidence_df["Confidence"].map(lambda x: f"{x:.1%}"),
                    color="Soil Type",
                )
                bar_fig.update_traces(textposition="outside")
                bar_fig.update_yaxes(range=[0, 1])
                style_plotly(bar_fig)
                st.plotly_chart(bar_fig, use_container_width=True)

            with col_chart_2:
                pie_fig = px.pie(
                    confidence_df,
                    names="Soil Type",
                    values="Confidence",
                    title="Confidence Share (Pie)",
                    hole=0.35,
                )
                style_plotly(pie_fig)
                st.plotly_chart(pie_fig, use_container_width=True)

            # Line chart helps visualize how sharply confidence drops after top class.
            line_fig = px.line(
                confidence_df.reset_index(drop=True).assign(Rank=lambda d: d.index + 1),
                x="Rank",
                y="Confidence",
                markers=True,
                title="Confidence Drop Across Ranked Classes (Line)",
            )
            line_fig.update_xaxes(dtick=1)
            line_fig.update_yaxes(range=[0, 1])
            style_plotly(line_fig)
            st.plotly_chart(line_fig, use_container_width=True)

            radar_fig = go.Figure()
            radar_fig.add_trace(
                go.Scatterpolar(
                    r=confidence_df["Confidence"].tolist(),
                    theta=confidence_df["Soil Type"].tolist(),
                    fill="toself",
                    name="Confidence Radar",
                )
            )
            radar_fig.update_layout(
                title="Confidence Profile (Radar)",
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                margin=dict(l=20, r=20, t=55, b=20),
            )
            st.plotly_chart(radar_fig, use_container_width=True)

            explain_prediction_block(input_df, confidence_df)

    if (
        hasattr(model, "named_steps")
        and "preprocessor" in model.named_steps
        and "classifier" in model.named_steps
        and hasattr(model.named_steps["classifier"], "feature_importances_")
    ):
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
        feature_importance_df = pd.DataFrame(
            {
                "Feature": preprocessor.get_feature_names_out(),
                "Importance": classifier.feature_importances_,
            }
        ).sort_values("Importance", ascending=False)
        st.markdown("#### Top Global Feature Importance")
        st.dataframe(feature_importance_df.head(10), use_container_width=True, hide_index=True)


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

    st.markdown("#### Batch Prediction Insights")
    class_count_df = (
        result_df["Predicted Soil Type"]
        .value_counts()
        .rename_axis("Soil Type")
        .reset_index(name="Count")
    )

    chart_col_1, chart_col_2 = st.columns(2)
    with chart_col_1:
        class_bar = px.bar(
            class_count_df,
            x="Soil Type",
            y="Count",
            title="Predicted Soil Type Distribution (Bar)",
            text="Count",
            color="Soil Type",
        )
        class_bar.update_traces(textposition="outside")
        style_plotly(class_bar)
        st.plotly_chart(class_bar, use_container_width=True)

    with chart_col_2:
        class_pie = px.pie(
            class_count_df,
            names="Soil Type",
            values="Count",
            title="Predicted Soil Type Distribution (Pie)",
            hole=0.35,
        )
        style_plotly(class_pie)
        st.plotly_chart(class_pie, use_container_width=True)

    if "Prediction Confidence" in result_df.columns:
        confidence_summary_df = (
            result_df.groupby("Predicted Soil Type", as_index=False)["Prediction Confidence"]
            .mean()
            .rename(columns={"Prediction Confidence": "Average Confidence"})
            .sort_values("Average Confidence", ascending=False)
        )
        conf_line = px.line(
            confidence_summary_df,
            x="Predicted Soil Type",
            y="Average Confidence",
            markers=True,
            title="Average Confidence by Predicted Class (Line)",
        )
        conf_line.update_yaxes(range=[0, 1])
        style_plotly(conf_line)
        st.plotly_chart(conf_line, use_container_width=True)

    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Full Results CSV",
        data=csv_buffer.getvalue(),
        file_name="soil_type_predictions.csv",
        mime="text/csv",
    )


def model_comparison_ui(selected_models):
    st.subheader("Model Comparison")
    st.caption("Compare benchmarked models using Accuracy, Macro F1, and training time.")

    benchmark_df = load_benchmark_results()
    if benchmark_df is None or benchmark_df.empty:
        st.info(
            "Benchmark file not found. Run `python benchmark_models.py --sample-size 40000` "
            "to generate `outputs/model_benchmark.json`."
        )
        return

    if selected_models:
        benchmark_df = benchmark_df[benchmark_df["model"].isin(selected_models)].copy()

    if benchmark_df.empty:
        st.warning("No benchmark rows match the selected models in the sidebar.")
        return

    st.dataframe(benchmark_df, use_container_width=True, hide_index=True)

    metric_col_1, metric_col_2 = st.columns(2)
    with metric_col_1:
        best_acc_row = benchmark_df.sort_values("accuracy", ascending=False).iloc[0]
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Best Accuracy Model</div><div class='metric-value'>{best_acc_row['model']}</div></div>",
            unsafe_allow_html=True,
        )
    with metric_col_2:
        best_f1_row = benchmark_df.sort_values("macro_f1", ascending=False).iloc[0]
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Best Macro F1 Model</div><div class='metric-value'>{best_f1_row['model']}</div></div>",
            unsafe_allow_html=True,
        )

    bar_acc = px.bar(
        benchmark_df.sort_values("accuracy", ascending=False),
        x="model",
        y="accuracy",
        color="model",
        title="Accuracy Comparison",
        text=benchmark_df.sort_values("accuracy", ascending=False)["accuracy"].map(lambda x: f"{x:.3f}"),
    )
    bar_acc.update_traces(textposition="outside")
    bar_acc.update_yaxes(range=[0, 1])
    style_plotly(bar_acc)
    st.plotly_chart(bar_acc, use_container_width=True)

    chart_col_1, chart_col_2 = st.columns(2)
    with chart_col_1:
        bar_f1 = px.bar(
            benchmark_df.sort_values("macro_f1", ascending=False),
            x="model",
            y="macro_f1",
            color="model",
            title="Macro F1 Comparison",
            text=benchmark_df.sort_values("macro_f1", ascending=False)["macro_f1"].map(lambda x: f"{x:.3f}"),
        )
        bar_f1.update_traces(textposition="outside")
        bar_f1.update_yaxes(range=[0, 1])
        style_plotly(bar_f1)
        st.plotly_chart(bar_f1, use_container_width=True)

    with chart_col_2:
        line_time = px.line(
            benchmark_df.sort_values("train_seconds", ascending=True),
            x="model",
            y="train_seconds",
            markers=True,
            title="Training Time Comparison (seconds)",
        )
        style_plotly(line_time)
        st.plotly_chart(line_time, use_container_width=True)


def main():
    st.set_page_config(page_title="Soil Type Classifier", page_icon="🌱", layout="wide")
    apply_custom_style()
    st.markdown("<div class='main-title'>Soil Type Classification Dashboard</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-title'>Interactive ML dashboard for soil type prediction with confidence and batch analytics.</div>",
        unsafe_allow_html=True,
    )

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

    with st.sidebar:
        st.header("Benchmarked Models")
        available_models = ["RandomForest", "ExtraTrees", "LogisticRegression", "LinearSVC"]
        selected_models = st.multiselect(
            "Select models to display",
            options=available_models,
            default=available_models,
        )
        st.caption(
            "Selected: "
            + (", ".join(selected_models) if selected_models else "None")
        )
        st.markdown("---")
        st.header("Project Overview")
        st.write("Primary Model: Random Forest")
        st.write(f"Classes: {', '.join(encoder.classes_)}")
        st.write("Features: Temparature, Humidity, Moisture, Crop Type, Nitrogen, Potassium, Phosphorous, ph")

    tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Model Comparison"])
    with tab1:
        single_prediction_ui(model, encoder)
    with tab2:
        batch_prediction_ui(model, encoder)
    with tab3:
        model_comparison_ui(selected_models)

    st.markdown("---")
    st.caption("Tip: Keep column names exactly as used in training, including 'Temparature'.")


if __name__ == "__main__":
    main()
