# ==========================================
# Lung Disease Classification Training Script
# Generator-based (with sample_weight)
# Output: lung_disease_model.h5
# ==========================================

import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# CONFIG
# ==========================================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20

BASE_PATH = "dataset/lung_train/train"

CLASS_NAMES = ['normal', 'covid', 'pneumonia', 'lung_opacity']
NUM_CLASSES = len(CLASS_NAMES)

# ==========================================
# COMPUTE CLASS WEIGHTS (ONCE)
# ==========================================
labels_for_weights = []

for idx, cls in enumerate(CLASS_NAMES):
    folder = os.path.join(BASE_PATH, "images", cls)
    labels_for_weights.extend([idx] * len(os.listdir(folder)))

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels_for_weights),
    y=labels_for_weights
)

CLASS_WEIGHT_MAP = dict(enumerate(class_weights))
print("Class Weights:", CLASS_WEIGHT_MAP)

# ==========================================
# DATA GENERATOR (IMAGE + MASK + SAMPLE_WEIGHT)
# ==========================================
def train_generator():
    images, labels, weights = [], [], []

    while True:
        for idx, cls in enumerate(CLASS_NAMES):
            img_dir = os.path.join(BASE_PATH, "images", cls)
            mask_dir = os.path.join(BASE_PATH, "masks", cls)

            for file in os.listdir(img_dir):
                img_path = os.path.join(img_dir, file)
                mask_path = os.path.join(mask_dir, file)

                if not os.path.exists(mask_path):
                    continue

                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path, 0)

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

                img = img / 255.0
                mask = mask / 255.0
                mask = np.expand_dims(mask, axis=-1)

                # APPLY LUNG MASK
                img = img * mask

                images.append(img)
                labels.append(idx)
                weights.append(CLASS_WEIGHT_MAP[idx])

                if len(images) == BATCH_SIZE:
                    yield (
                        np.array(images),
                        tf.keras.utils.to_categorical(labels, NUM_CLASSES),
                        np.array(weights)
                    )
                    images, labels, weights = [], [], []

# ==========================================
# MODEL (MobileNetV2)
# ==========================================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================
# TRAINING
# ==========================================
train_gen = train_generator()

total_images = sum(
    len(os.listdir(os.path.join(BASE_PATH, "images", c)))
    for c in CLASS_NAMES
)

steps_per_epoch = total_images // BATCH_SIZE

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS
)

# ==========================================
# SAVE MODEL
# ==========================================
model.save("lung_disease_model.h5")
print("✅ lung_disease_model.h5 saved successfully")
