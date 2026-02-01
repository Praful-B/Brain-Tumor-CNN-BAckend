import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# 
# CONFIG
# 

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

TRAIN_DIR = "model/dataset/train"
VAL_DIR = "model/dataset/val"
TEST_DIR = "model/dataset/test"

# 
# LOAD DATASETS
# 

train_ds = image_dataset_from_directory(TRAIN_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, color_mode="rgb", shuffle=True)

CLASS_NAMES = train_ds.class_names


val_ds = image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    shuffle=False
)

test_ds = image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    shuffle=False
)


# NORMALIZATION


normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))


# PERFORMANCE OPTIMIZATION


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

print("Datasets loaded and preprocessed successfully.")

__all__ = ["train_ds", "val_ds", "test_ds", "CLASS_NAMES"]
