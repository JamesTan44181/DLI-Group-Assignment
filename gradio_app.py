# gradio_app.py
import os, io, base64, re
import numpy as np
import pandas as pd
from typing import List, Optional

# ---- Keras loader (TF-Keras first, Keras3 fallback) ----
def load_model_any(path):
    try:
        from tensorflow.keras.models import load_model as tf_load
        return tf_load(path)
    except Exception as e1:
        try:
            from keras.models import load_model as k_load
            return k_load(path)
        except Exception as e2:
            raise RuntimeError(f"Failed to load model:\nTF: {e1}\nKeras3: {e2}")

# ---- Optional scaler via joblib ----
try:
    import joblib
except Exception:
    joblib = None

import gradio as gr

# -------- Config / Auto-discovery --------
CANDIDATE_MODEL_PATHS = [
    "/content/drive/MyDrive/DLI-assignment/phishing_model.keras",
    "/content/drive/MyDrive/Colab Notebooks/phishing_model.keras",
    "/content/DLI-Group-Assignment/phishing_model.keras",
    "/content/DLI-Group-Assignment/model.h5",
    "./phishing_model.keras",
    "./model.h5",
]
CANDIDATE_XTEST_PATHS = [
    "/content/drive/MyDrive/DLI-assignment/X_test.csv",
    "/content/drive/MyDrive/Colab Notebooks/X_test.csv",
    "/content/DLI-Group-Assignment/X_test.csv",
    "./X_test.csv",
]
CANDIDATE_DATASET_PATHS = [
    "/content/DLI-Group-Assignment/cleaned_balanced_dataset.csv",
    "./cleaned_balanced_dataset.csv",
]
CANDIDATE_SCALERS = [
    "/content/drive/MyDrive/DLI-assignment/scaler.joblib",
    "/content/drive/MyDrive/Colab Notebooks/scaler.joblib",
    "/content/DLI-Group-Assignment/scaler.joblib",
    "./scaler.joblib",
]

def find_first(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None

MODEL_PATH  = find_first(CANDIDATE_MODEL_PATHS)
XTEST_PATH  = find_first(CANDIDATE_XTEST_PATHS)
DATA_PATH   = find_first(CANDIDATE_DATASET_PATHS)
SCALER_PATH = find_first(CANDIDATE_SCALERS)

if not MODEL_PATH:
    raise FileNotFoundError("Model not found. Put phishing_model.keras in repo or in MyDrive/DLI-assignment.")

model = load_model_any(MODEL_PATH)
scaler = joblib.load(SCALER_PATH) if (SCALER_PATH and joblib) else None

# -------- Columns (order matters) --------
def expected_cols() -> List[str]:
    if XTEST_PATH and os.path.exists(XTEST_PATH):
        return list(pd.read_csv(XTEST_PATH, nrows=5).columns)
    if DATA_PATH and os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, nrows=5)
        label_like = {"label","class","target","is_phishing","phishing"}
        return [c for c in df.columns if c not in label_like]
    return []
COLS = expected_cols()

# -------- Preprocess & predict --------
def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    if COLS:
        for c in COLS:
            if c not in df.columns: df[c] = 0
        df = df[COLS]
    df = ensure_numeric(df).fillna(0)
    if scaler is not None:
        try:
            df = pd.DataFrame(scaler.transform(df.values), columns=df.columns)
        except Exception:
            pass
    return df

def predict_batch(file, threshold=0.5):
    if file is None:
        return "Please upload a CSV file.", None, None
    try:
        df = pd.read_csv(file.name if hasattr(file, "name") else file)
    except Exception:
        file.seek(0); df = pd.read_csv(io.BytesIO(file.read()))
    feats = preprocess(df.copy())
    probs = model.predict(feats.values, verbose=0).ravel()
    out = df.copy()
    out["phishing_prob"] = probs
    out["prediction"] = (probs >= float(threshold)).astype(int)
    preview = out.head(20)

    tmp = "predictions.csv"
    out.to_csv(tmp, index=False)
    return preview, tmp, f"Rows: {len(out)} | Threshold: {threshold:.2f}"

def predict_single(row_df, threshold=0.5):
    if row_df is None or getattr(row_df, "shape", (0,0))[0] == 0:
        return "Provide exactly one row of features."
    row_df = pd.DataFrame(row_df)
    if COLS and list(row_df.columns) != COLS:
        row_df.columns = COLS[:row_df.shape[1]]
        for c in COLS[row_df.shape[1]:]: row_df[c] = 0
        row_df = row_df[COLS]
    feats = preprocess(row_df.copy())
    p = float(model.predict(feats.values, verbose=0).ravel()[0])
    pred = int(p >= float(threshold))
    return f"Phishing probability: **{p:.4f}** → Predicted label: **{pred}** (threshold={threshold:.2f})"

def template_df():
    return pd.DataFrame({c:[0] for c in COLS}) if COLS else pd.DataFrame([[0]], columns=["feature1"])

# -------- UI --------
model_status  = f"Model: `{MODEL_PATH}`"
scaler_status = f"Scaler: {'loaded' if scaler is not None else 'not found'}"
cols_status   = f"Columns detected: {len(COLS)}" if COLS else "Columns unknown (upload CSV with headers)."

with gr.Blocks(title="Phishing Detector GUI") as demo:
    gr.Markdown(f"## Phishing Detector (Deep Learning)\n{model_status} • {scaler_status} • {cols_status}")

    with gr.Tabs():
        with gr.Tab("Batch CSV Prediction"):
            csv_in = gr.File(label="Upload CSV with feature columns", file_types=[".csv"])
            thr1   = gr.Slider(0.05, 0.95, value=0.5, step=0.01, label="Decision threshold")
            run    = gr.Button("Predict")
            prev   = gr.Dataframe(label="Preview (first 20 rows)")
            dl     = gr.File(label="Download predictions.csv")
            meta   = gr.Markdown()
            run.click(predict_batch, inputs=[csv_in, thr1], outputs=[prev, dl, meta])

        with gr.Tab("Single Row"):
            gr.Markdown("Enter a single row (columns auto-detected from `X_test.csv` if available).")
            row    = gr.Dataframe(value=template_df(), row_count=(1, "fixed"), wrap=True)
            thr2   = gr.Slider(0.05, 0.95, value=0.5, step=0.01, label="Decision threshold")
            go     = gr.Button("Predict")
            out    = gr.Markdown()
            go.click(predict_single, inputs=[row, thr2], outputs=[out])

    gr.Markdown("> Tip: For best results, ensure your CSV columns match the training order.\n"
                "If `X_test.csv` is present, that order is used automatically.")

if __name__ == "__main__":
    demo.launch(share=True)
