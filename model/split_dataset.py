import os
import random
import shutil


# CONFIGURATION


RAW_DIR = "model/dataset/raw"
TRAIN_DIR = "model/dataset/train"
VAL_DIR = "model/dataset/val"
TEST_DIR = "model/dataset/test"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# INITIAL SETUP


random.seed(RANDOM_SEED)

classes = os.listdir(RAW_DIR)

for cls in classes:
    os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, cls), exist_ok=True)


# SPLITTING LOGIC


for cls in classes:
    cls_path = os.path.join(RAW_DIR, cls)
    images = os.listdir(cls_path)

    random.shuffle(images)

    total_images = len(images)
    train_end = int(total_images * TRAIN_RATIO)
    val_end = train_end + int(total_images * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for img in train_images:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(TRAIN_DIR, cls, img)
        )

    for img in val_images:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(VAL_DIR, cls, img)
        )

    for img in test_images:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(TEST_DIR, cls, img)
        )

    print(
        f"{cls}: "
        f"{len(train_images)} train, "
        f"{len(val_images)} val, "
        f"{len(test_images)} test"
    )

print("\nDataset split completed successfully.")
