import os
import cv2
import json
import uuid
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# -------------------------------------------------
# Paths
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "model",
    "trained_multi_class_model.keras"
)

CLASS_INDEX_PATH = os.path.join(
    PROJECT_ROOT,
    "model",
    "class_indices.json"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "storage",
    "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------
# Lazy model loading (CRITICAL)
# -------------------------------------------------
_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
    return _model

# -------------------------------------------------
# Load class names safely
# -------------------------------------------------
with open(CLASS_INDEX_PATH, "r") as f:
    class_indices = json.load(f)

CLASS_NAMES = {v: k for k, v in class_indices.items()}

# -------------------------------------------------
# Preprocessing
# -------------------------------------------------
def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# -------------------------------------------------
# Grad-CAM core
# -------------------------------------------------
def generate_gradcam(model, img_array, class_index):
    # Use LAST convolutional layer safely
    last_conv_layer = None
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        raise RuntimeError("No conv layer found for Grad-CAM")

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    return heatmap.numpy()

# -------------------------------------------------
# Overlay heatmap
# -------------------------------------------------
def overlay_heatmap(original_path, heatmap):
    img = cv2.imread(original_path)
    img = cv2.resize(img, (224, 224))

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    filename = f"{uuid.uuid4().hex}.png"
    output_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(output_path, superimposed)

    return filename

# -------------------------------------------------
# Public API used by Flask
# -------------------------------------------------
def run_gradcam(image_path):
    model = get_model()
    img_array = preprocess_image(image_path)

    preds = model.predict(img_array)[0]
    class_index = int(np.argmax(preds))

    label = CLASS_NAMES[class_index]
    confidence = float(preds[class_index]) * 100

    heatmap = generate_gradcam(model, img_array, class_index)
    filename = overlay_heatmap(image_path, heatmap)

    return filename, label, confidence
