import os
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from app.model.gradcam import run_gradcam

# -------------------------------------------------
# App setup
# -------------------------------------------------
app = Flask(__name__)

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

UPLOAD_DIR = os.path.join(PROJECT_ROOT, "storage", "uploads")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "storage", "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "Empty filename", 400

    filename = secure_filename(file.filename)
    upload_path = os.path.join(UPLOAD_DIR, filename)
    file.save(upload_path)

    gradcam_file, label, confidence = run_gradcam(upload_path)

    return render_template(
        "result.html",
        label=label,
        confidence=round(confidence, 2),
        gradcam_filename=gradcam_file
    )

@app.route("/outputs/<filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_DIR, filename)

# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
