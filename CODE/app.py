# app.py
import os
import io
import shutil
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, flash
from werkzeug.utils import secure_filename

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa

# -----------------------------
# Configuration
# -----------------------------
APP_ROOT = Path(__file__).parent.resolve()
MODEL_DIR = APP_ROOT / "model"
UPLOAD_DIR = APP_ROOT / "uploads"
STATIC_CHARTS = APP_ROOT / "static" / "charts"

MODEL_FILE = MODEL_DIR / "yamnet_gru_smote_tuned.keras"
THRESH_FILE = MODEL_DIR / "best_threshold.txt"

ALLOWED_EXT = {".wav", ".flac", ".ogg", ".mp3"}
MAX_SEQ = 100   # must match training
TARGET_SR = 16000

# create directories if missing
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATIC_CHARTS.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Flask app init
# -----------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "replace_this_with_a_random_secret"

# Temporary in-memory user store (no DB)
_users = {}

# -----------------------------
# Load ML assets at startup
# -----------------------------
# Load Keras model
from keras.saving import load_model  # important fix for Keras 3

try:
    model = load_model(
        str(MODEL_FILE),
        compile=False,
        safe_mode=False
    )
    print("[INFO] Model loaded successfully with custom loader.")
except Exception as e:
    print("[ERROR] Failed to load model:", e)
    model = None


# Load threshold
if not THRESH_FILE.exists():
    print(f"[WARN] Threshold file not found at {THRESH_FILE}. Using default 0.5")
    best_threshold = 0.5
else:
    try:
        best_threshold = float(THRESH_FILE.read_text().strip())
    except Exception as e:
        print("[WARN] Failed to read threshold file:", e)
        best_threshold = 0.5
print(f"[INFO] Using threshold = {best_threshold}")

# Load yamnet from TF Hub once
print("[INFO] Loading YAMNet from TF Hub (this may take a few seconds)...")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
print("[INFO] YAMNet loaded.")

# -----------------------------
# Utility functions
# -----------------------------
def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXT

def load_audio_file(path, target_sr=TARGET_SR):
    """Load audio and return 1D float32 waveform at target_sr."""
    audio, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)

def yamnet_sequence_embeddings(waveform):
    """Return numpy array of YAMNet embeddings (T, 1024) for a 1D waveform."""
    waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)
    scores, embeddings, spectrogram = yamnet(waveform_tf)
    return embeddings.numpy()

def pad_sequence(seq, max_len=MAX_SEQ):
    """Pad or trim a (T, D) sequence to (max_len, D)."""
    T, D = seq.shape
    if T >= max_len:
        return seq[:max_len]
    else:
        pad_len = max_len - T
        pad = np.zeros((pad_len, D), dtype=np.float32)
        return np.vstack([seq, pad])

def extract_features_from_path(filepath):
    """Load audio, extract yamnet embeddings, pad to MAX_SEQ and return shape (1, MAX_SEQ, 1024)."""
    waveform = load_audio_file(str(filepath))
    emb = yamnet_sequence_embeddings(waveform)   # (T, 1024)
    emb_pad = pad_sequence(emb, max_len=MAX_SEQ) # (MAX_SEQ, 1024)
    return np.expand_dims(emb_pad, axis=0)       # (1, MAX_SEQ, 1024)

# -----------------------------
# Routes: Home / Auth / Pages
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        uname = request.form.get("username", "").strip()
        pwd = request.form.get("password", "").strip()
        if not uname or not pwd:
            flash("Username and password required.", "warning")
            return redirect(url_for("register"))
        if uname in _users:
            flash("User already exists. Choose another username.", "danger")
            return redirect(url_for("register"))
        _users[uname] = pwd
        flash("Registration successful. Please login.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        uname = request.form.get("username", "").strip()
        pwd = request.form.get("password", "").strip()
        if uname in _users and _users[uname] == pwd:
            session["user"] = uname
            flash(f"Welcome, {uname}!", "success")
            return redirect(url_for("predict"))
        flash("Invalid credentials.", "danger")
        return redirect(url_for("login"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out.", "info")
    return redirect(url_for("index"))


# -----------------------------
# Prediction page
# -----------------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        flash("Please log in to use prediction.", "warning")
        return redirect(url_for("login"))

    if request.method == "POST":
        if "audio" not in request.files:
            flash("No file part in the request.", "danger")
            return redirect(request.url)

        file = request.files["audio"]
        if file.filename == "":
            flash("No selected file.", "warning")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Unsupported file type. Allowed: wav, ogg, flac, mp3", "danger")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        save_path = UPLOAD_DIR / filename
        file.save(save_path)

        try:
            # extract features
            X = extract_features_from_path(save_path)  # (1, MAX_SEQ, 1024)
            if model is None:
                flash("Model is not loaded. Please place the model in model/ folder.", "danger")
                return redirect(request.url)

            # predict
            prob = float(model.predict(X, verbose=0).ravel()[0])
            label = "Abnormal" if prob >= best_threshold else "Normal"

            # clean up upload
            try:
                save_path.unlink(missing_ok=True)
            except Exception:
                pass

            return render_template("result.html", filename=filename, prob=prob, label=label, threshold=best_threshold)
        except Exception as e:
            # keep file for debugging (or remove depending on your preference)
            flash(f"Error during prediction: {e}", "danger")
            app.logger.exception("Prediction error")
            return redirect(request.url)

    return render_template("predict.html")


# -----------------------------
# Charts page (serves static images if present)
# -----------------------------
@app.route("/charts")
def charts():
    # list chart image files in static/charts
    chart_files = []
    if STATIC_CHARTS.exists():
        for f in STATIC_CHARTS.iterdir():
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}:
                chart_files.append(f.name)
    return render_template("charts.html", charts=chart_files)


@app.route("/static/charts/<path:filename>")
def send_chart(filename):
    return send_from_directory(str(STATIC_CHARTS), filename)


# -----------------------------
# Simple health check
# -----------------------------
@app.route("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "threshold": best_threshold}, 200


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(host="0.0.0.0", port=5000, debug=True)
