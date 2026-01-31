import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ------------------ PATH SETUP ------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# Change the import to use the correct path
try:
    from app.utils.brain_mask import get_brain_mask
except ImportError:
    # Fallback if running standalone
    from utils.brain_mask import get_brain_mask

# ------------------ CONFIG ------------------
IMG_SIZE = 224
# Update model path to point to the root directory
MODEL_PATH = os.path.join(PROJECT_ROOT, "..", "trained_multi_class_model.keras")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "storage", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# ------------------ LOAD MODEL ------------------
model = load_model(MODEL_PATH)

# Warm-up (prevents TF shape issues)
_ = model(tf.zeros((1, IMG_SIZE, IMG_SIZE, 3)))

# ------------------ PREPROCESS ------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_input = img_resized / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    return img, img_input

# ------------------ GRAD-CAM ------------------
def generate_gradcam(img_input, class_index):
    last_conv_layer = model.get_layer("out_relu")

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_input, training=False)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

# ------------------ MASK + OVERLAY ------------------
def apply_brain_mask(heatmap, original_img):
    mask = get_brain_mask(original_img)
    mask = cv2.resize(mask, (heatmap.shape[1], heatmap.shape[0]))
    mask = mask / 255.0
    return heatmap * mask

def overlay_heatmap(original_img, heatmap):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )
    return cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)

# ------------------ SAVE RESULT (NO POPUP) ------------------
def save_result(image, label, confidence, save_path):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.title(
        f"Prediction: {label}\nConfidence: {confidence:.2f}%",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()   # ❗ NO plt.show()

# ------------------ PIPELINE ------------------
def run_gradcam(image_path):
    original_img, img_input = preprocess_image(image_path)

    preds = model.predict(img_input)[0]
    class_index = int(np.argmax(preds))

    confidence = float(preds[class_index]) * 100   # percent ONCE
    label = CLASS_NAMES[class_index]

    heatmap = generate_gradcam(img_input, class_index)
    heatmap = apply_brain_mask(heatmap, original_img)
    result = overlay_heatmap(original_img, heatmap)

    filename = f"gradcam_{os.path.basename(image_path)}"
    save_path = os.path.join(OUTPUT_DIR, filename)

    save_result(result, label, confidence, save_path)

    # 🔑 RETURN RELATIVE PATH FOR FASTAPI
    return f"storage/outputs/{filename}", label, confidence

# ------------------ TEST ------------------
if __name__ == "__main__":
    image_path = input("Enter full path to MRI image: ").strip()

    if not os.path.exists(image_path):
        print("❌ Invalid path")
    else:
        path, label, conf = run_gradcam(image_path)
        print("\n✅ Result")
        print("Saved at:", path)
        print("Label:", label)
        print("Confidence:", round(conf, 2), "%")
