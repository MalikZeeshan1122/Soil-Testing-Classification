import argparse
import pandas as pd
import joblib


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict Soil Type using a trained Random Forest model."
    )
    parser.add_argument("--Temparature", type=float, required=True)
    parser.add_argument("--Humidity", type=float, required=True)
    parser.add_argument("--Moisture", type=float, required=True)
    parser.add_argument("--Crop_Type", type=str, required=True)
    parser.add_argument("--Nitrogen", type=float, required=True)
    parser.add_argument("--Potassium", type=float, required=True)
    parser.add_argument("--Phosphorous", type=float, required=True)
    parser.add_argument("--ph", type=float, required=True)
    parser.add_argument(
        "--model-path", default="soil_rf_model.joblib", help="Path to saved model file"
    )
    parser.add_argument(
        "--encoder-path",
        default="soil_label_encoder.joblib",
        help="Path to saved label encoder file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model = joblib.load(args.model_path)
    label_encoder = joblib.load(args.encoder_path)

    sample = pd.DataFrame(
        [
            {
                "Temparature": args.Temparature,
                "Humidity": args.Humidity,
                "Moisture": args.Moisture,
                "Crop Type": args.Crop_Type,
                "Nitrogen": args.Nitrogen,
                "Potassium": args.Potassium,
                "Phosphorous": args.Phosphorous,
                "ph": args.ph,
            }
        ]
    )

    pred_idx = model.predict(sample)[0]
    pred_soil_type = label_encoder.inverse_transform([pred_idx])[0]

    print("Predicted Soil Type:", pred_soil_type)


if __name__ == "__main__":
    main()
