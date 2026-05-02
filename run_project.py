import argparse
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
BENCHMARK_SCRIPT = BASE_DIR / "benchmark_models.py"


def run_cmd(args):
    result = subprocess.run(args, cwd=BASE_DIR)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def parse_args():
    parser = argparse.ArgumentParser(
        description="One-command flow: train (if needed) -> sample prediction -> optional benchmark -> Streamlit app."
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark_models.py on the full dataset before launching the app.",
    )
    parser.add_argument(
        "--skip-app",
        action="store_true",
        help="Do not launch Streamlit at the end (useful for CI / scripted runs).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

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
            "--Fertilizer_Name",
            "Ammonium Phosphate Complex",
            "--Soil_pH_Type",
            "Alkaline",
        ]
    )

    if args.benchmark:
        if not BENCHMARK_SCRIPT.exists():
            print("\nbenchmark_models.py not found. Skipping benchmark step.")
        else:
            print("\nRunning full-dataset benchmark (this may take several minutes)...")
            run_cmd([sys.executable, str(BENCHMARK_SCRIPT), "--sample-size", "0"])

    if args.skip_app:
        print("\n--skip-app set. Not launching Streamlit.")
        return

    print("\nStarting Streamlit app...")
    print("Tip: first launch may take a few seconds. Use Ctrl+C to stop the app.")
    try:
        run_cmd([sys.executable, "-m", "streamlit", "run", str(APP_SCRIPT)])
    except KeyboardInterrupt:
        print("\nStreamlit app stopped by user.")


if __name__ == "__main__":
    main()
