import os
import numpy as np
import pydicom
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# ================= CONFIG =================
DATASET_DIR = "DICOM_CT_Brain"
IMG_SIZE = 128
EPOCHS = 10
BATCH_SIZE = 16

# ================= LOAD DATA =================
X = []
y = []

def load_dicom_image(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.stack([img, img, img], axis=-1)  # convert to 3-channel
    return img / 255.0

for label, folder in enumerate(["no", "yes"]):
    folder_path = os.path.join(DATASET_DIR, folder)

    for file in os.listdir(folder_path):
        if file.endswith(".dcm"):
            img_path = os.path.join(folder_path, file)
            img = load_dicom_image(img_path)
            X.append(img)
            y.append(label)

X = np.array(X)
y = to_categorical(y, num_classes=2)

print("Dataset loaded:", X.shape, y.shape)

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# ================= CNN MODEL =================
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(2, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ================= TRAIN =================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# ================= SAVE =================
model.save("models/ct_brain_cnn.h5")
print("✅ CT Brain CNN Model saved as ct_brain_cnn.h5")
