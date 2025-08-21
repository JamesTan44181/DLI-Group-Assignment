# gradio_app.py
# GUI for your prototype: loads trained Keras model, supports CSV batch + single-row prediction.
# Works great in Colab (launches with share=True).

import os, io, base64
import numpy as np
import pandas as pd
from typing import List, Optional
from tensorflow.keras.models import load_model

try:
    import joblib
except Exception:
    joblib = None

import gradio as gr

# -------- Config / Auto-discovery --------
CANDIDATE_MODEL_PATHS = [
    "/content/drive/MyDrive/Colab Notebooks/phishing_model.keras",
    "/content/DLI-Group-Assignment/phishing_model.keras",
    "/content/DLI-Group-Assignment/model.h5",
    "./phishing_model.keras",
    "./model.h5",
]

CANDIDATE_XTEST_PATHS = [
    "/content/drive/MyDrive/Colab Notebooks/X_test.csv",
    "/content/DLI-Group-Assignment/X_test.csv",
    "./X_test.csv",
]

CANDIDATE_DATASET_PATHS = [
    "/content/DLI-Group-Assignment/cleaned_balanced_dataset.csv",
    "./cleaned_balanced_dataset.csv",
]

CANDIDATE_SCALERS = [
    "/content/drive/MyDrive/Colab Notebooks/scaler.joblib",
    "/content/drive/MyDrive/Colab Notebooks/scaler.pkl",
    "/content/DLI-Group-Assignment/scaler.joblib",
    "/content/DLI-Group-Assignment/scaler.pkl",
    "./scaler.joblib",
    "./scaler.pkl",
]


def find_first(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


MODEL_PATH = find_first(CANDIDATE_MODEL_PATHS)
XTEST_PATH = find_first(CANDIDATE_XTEST_PATHS)
DATASET_PATH = find_first(CANDIDATE_DATASET_PATHS)
SCALER_PATH = find_first(CANDIDATE_SCALERS)

# -------- Load model & scaler --------
if MODEL_PATH is None:
    raise FileNotFoundError(
        "No trained model found. Place 'phishing_model.keras' (or model.h5) in the repo "
        "or in Drive: /content/drive/MyDrive/Colab Notebooks/"
    )
model = load_model(MODEL_PATH)

scaler = None
if SCALER_PATH and joblib:
    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception:
        scaler = None

# -------- Determine expected columns (order matters) --------
def derive_expected_cols() -> List[str]:
    # Prefer X_test.csv columns (exact training order)
    if XTEST_PATH and os.path.exists(XTEST_PATH):
        df = pd.read_csv(XTEST_PATH, nrows=5)
        return list(df.columns)

    # Fallback: dataset columns minus common label names
    if DATASET_PATH and os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH, nrows=5)
        cols = list(df.columns)
        label_like = {"label", "class", "target", "is_phishing", "phishing"}
        return [c for c in cols if c not in label_like]

    # Last resort: ask user to upload CSV with header
    return []

EXPECTED_COLS = derive_expected_cols()

# -------- Preprocess --------
def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    if EXPECTED_COLS:
        # Add missing columns as 0, keep only expected in right order
        for c in EXPECTED_COLS:
            if c not in df.columns:
                df[c] = 0
        df = df[EXPECTED_COLS]
    df = ensure_numeric(df)
    df = df.fillna(0)
    if scaler is not None:
        try:
            df = pd.DataFrame(scaler.transform(df.values), columns=df.columns)
        except Exception:
            # If scaler shape mismatches, skip scaling
            pass
    return df

# -------- Prediction helpers --------
def predict_proba_array(df: pd.DataFrame) -> np.ndarray:
    preds = model.predict(df.values, verbose=0).ravel()
    return preds  # probability of phishing (positive class)

def predict_batch(file, threshold=0.5):
    """
    file: CSV file with header containing feature columns
    """
    if file is None:
        return "Please upload a CSV file.", None, None

    try:
        df = pd.read_csv(file.name if hasattr(file, "name") else file)
    except Exception:
        # Fallback read from bytes
        file.seek(0)
        df = pd.read_csv(io.BytesIO(file.read()))

    features = preprocess(df.copy())
    probs = predict_proba_array(features)
    out = df.copy()
    out["phishing_prob"] = probs
    out["prediction"] = (probs >= float(threshold)).astype(int)

    # Return a small preview and a downloadable CSV
    preview = out.head(20)
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv_bytes).decode()
    download_link = f"data:text/csv;base64,{b64}"

    return preview, download_link, f"Rows: {len(out)} | Threshold: {threshold:.2f}"

def predict_single(row_df: pd.DataFrame, threshold=0.5):
    """
    row_df: gr.Dataframe with exactly 1 row and feature columns
    """
    if row_df is None or row_df.shape[0] == 0:
        return "Provide one row of features."
    row_df = pd.DataFrame(row_df)
    # If user didn't get headers in, fix with EXPECTED_COLS
    if EXPECTED_COLS and list(row_df.columns) != EXPECTED_COLS:
        row_df.columns = EXPECTED_COLS[: row_df.shape[1]]
        # If shorter, pad missing columns
        for c in EXPECTED_COLS[row_df.shape[1]:]:
            row_df[c] = 0
        row_df = row_df[EXPECTED_COLS]

    feats = preprocess(row_df.copy())
    p = float(predict_proba_array(feats)[0])
    pred = int(p >= float(threshold))
    return f"Phishing probability: {p:.4f}  →  Predicted label: {pred} (threshold={threshold:.2f})"

# -------- Gradio UI --------
def build_df_template():
    # Build a 1-row template with zeros and reasonable defaults.
    if EXPECTED_COLS:
        data = {c: [0] for c in EXPECTED_COLS}
        return pd.DataFrame(data)
    # Otherwise show an empty table and ask user to upload CSV in Batch tab
    return pd.DataFrame([[0]], columns=["feature1"])

model_status = f"Model loaded from: {MODEL_PATH}"
scaler_status = f"Scaler: {'loaded' if scaler is not None else 'not found'}"
cols_status = f"Columns: {len(EXPECTED_COLS)} detected" if EXPECTED_COLS else "Columns: unknown (upload CSV with header)"

with gr.Blocks(title="Phishing Detector GUI") as demo:
    gr.Markdown(f"### Phishing Detector (Deep Learning)\n**{model_status}**  •  **{scaler_status}**  •  **{cols_status}**")

    with gr.Tabs():
        with gr.Tab("Batch CSV Prediction"):
            csv_input = gr.File(label="Upload CSV with feature columns", file_types=[".csv"])
            th1 = gr.Slider(0.05, 0.95, value=0.5, step=0.01, label="Decision threshold")
            run_btn = gr.Button("Predict on CSV")
            preview = gr.Dataframe(label="Preview of predictions (first 20 rows)")
            dl = gr.File(label="Download predictions CSV", visible=False)
            meta = gr.Markdown()

            def _batch(file, threshold):
                prev, link, info = predict_batch(file, threshold)
                # Emit a temp CSV file so Gradio shows a download
                tmp_path = "predictions.csv"
                if isinstance(prev, str):
                    return prev, None, info
                pd.DataFrame(prev).to_csv(tmp_path, index=False)
                return prev, tmp_path, info

            run_btn.click(_batch, inputs=[csv_input, th1], outputs=[preview, dl, meta])

        with gr.Tab("Single Prediction (1 row)"):
            gr.Markdown("Enter one row of features (columns auto-detected if possible).")
            template = gr.Dataframe(value=build_df_template(), row_count=(1, "fixed"), wrap=True)
            th2 = gr.Slider(0.05, 0.95, value=0.5, step=0.01, label="Decision threshold")
            single_btn = gr.Button("Predict")
            single_out = gr.Markdown()
            single_btn.click(predict_single, inputs=[template, th2], outputs=[single_out])

    gr.Markdown("> Tip: For best results, ensure your CSV columns match the training order.\
 If X_test.csv is present, the app uses its column order automatically.")

if __name__ == "__main__":
    demo.launch(share=True)  # produces a public URL; perfect for submission screenshots

