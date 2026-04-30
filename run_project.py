import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Soil sample testing.csv"
MODEL_FILE = BASE_DIR / "soil_rf_model.joblib"
ENCODER_FILE = BASE_DIR / "soil_label_encoder.joblib"
TRAIN_SCRIPT = BASE_DIR / "train_rf.py"
PREDICT_SCRIPT = BASE_DIR / "predict_rf.py"
APP_SCRIPT = BASE_DIR / "app.py"


def run_cmd(args):
    result = subprocess.run(args, cwd=BASE_DIR)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    if not DATA_FILE.exists():
        raise SystemExit(f"Dataset not found: {DATA_FILE}")

    if not TRAIN_SCRIPT.exists() or not PREDICT_SCRIPT.exists() or not APP_SCRIPT.exists():
        raise SystemExit("One or more required scripts are missing (train_rf.py, predict_rf.py, app.py).")

    if not MODEL_FILE.exists() or not ENCODER_FILE.exists():
        print("Model artifacts not found. Training model first...")
        run_cmd([sys.executable, str(TRAIN_SCRIPT)])
    else:
        print("Model artifacts found. Skipping training.")

    print("\nRunning sample prediction check...")
    run_cmd(
        [
            sys.executable,
            str(PREDICT_SCRIPT),
            "--Temparature",
            "30",
            "--Humidity",
            "55",
            "--Moisture",
            "45",
            "--Crop_Type",
            "Ground Nuts",
            "--Nitrogen",
            "8",
            "--Potassium",
            "4",
            "--Phosphorous",
            "18",
            "--ph",
            "7.2",
        ]
    )

    print("\nStarting Streamlit app...")
    run_cmd([sys.executable, "-m", "streamlit", "run", str(APP_SCRIPT)])


if __name__ == "__main__":
    main()
