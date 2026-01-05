import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "lung")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

img_size = (224, 224)
batch_size = 8

# Data generator with train/validation split
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="validation"
)

# Simple CNN model
model = models.Sequential([
    layers.Input(shape=(*img_size, 3)),
    layers.Conv2D(16, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("Starting training...")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5
)

# Save model
model_path = os.path.join(MODELS_DIR, "lung_model.h5")
model.save(model_path)
print(f"Model saved to {model_path}")
print("Classes mapping:", train_gen.class_indices)
