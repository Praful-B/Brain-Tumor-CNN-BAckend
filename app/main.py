from flask import Flask, render_template, request, jsonify, url_for
import os
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw
import time

# -----------------------
# PATH CONFIG
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_DIR = os.path.join(BASE_DIR, "static")
GRADCAM_DIR = os.path.join(STATIC_DIR, "gradcam")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)

# -----------------------
# FLASK APP
# -----------------------
app = Flask(
    __name__,
    template_folder=TEMPLATES_DIR,
    static_folder=STATIC_DIR
)

app.config["UPLOAD_FOLDER"] = UPLOAD_DIR

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# -----------------------
# ROUTES
# -----------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(image_path)

    # -----------------------
    # 🔥 MODEL OUTPUT (DEMO)
    # -----------------------
    prediction = "Glioma"
    confidence = 0.92

    gradcam_url = None

    if prediction != "No Tumor":
        # UNIQUE filename to avoid browser cache
        ts = int(time.time())
        gradcam_filename = f"gradcam_{ts}.png"
        gradcam_path = os.path.join(GRADCAM_DIR, gradcam_filename)

        # 🔴 CREATE A REAL IMAGE (no placeholders)
        img = Image.open(image_path).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (255, 0, 0, 90))

        draw = ImageDraw.Draw(overlay)
        w, h = img.size
        draw.ellipse(
            (w*0.3, h*0.3, w*0.7, h*0.7),
            fill=(255, 0, 0, 160)
        )

        gradcam_img = Image.alpha_composite(img, overlay)
        gradcam_img.save(gradcam_path)

        gradcam_url = url_for("static", filename=f"gradcam/{gradcam_filename}")

    return jsonify({
        "prediction": prediction,
        "confidence": confidence,
        "gradcam": gradcam_url
    })


# -----------------------
# RUN
# -----------------------
if __name__ == "__main__":
    app.run(debug=True)
