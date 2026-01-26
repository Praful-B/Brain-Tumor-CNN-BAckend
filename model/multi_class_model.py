import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =========================
# Configuration
# =========================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "model", "dataset")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

MODEL_SAVE_PATH = os.path.join(
    PROJECT_ROOT, "model", "trained_multi_class_model.keras"
)

# =========================
# Data Generators
# =========================
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# 🔴 IMPORTANT: detect classes automatically
NUM_CLASSES = train_data.num_classes
print("Detected classes:", train_data.class_indices)

# =========================
# Model
# =========================
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

base_model = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_tensor=inputs
)

base_model.trainable = False  # transfer learning

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# Train
# =========================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# =========================
# Save Model
# =========================
model.save(MODEL_SAVE_PATH)
print("Model saved to:", MODEL_SAVE_PATH)
