import os
import cv2
import numpy as np
from unet_model import unet
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 256
IMG_DIR = "dataset/images/flair"
MASK_DIR = "dataset/masks"

X, Y = [], []

for img_name in os.listdir(IMG_DIR):
    img = cv2.imread(os.path.join(IMG_DIR, img_name))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    mask_name = img_name.replace("flair", "mask")
    mask = cv2.imread(os.path.join(MASK_DIR, mask_name), 0)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    mask = mask / 255.0
    mask = np.expand_dims(mask, axis=-1)

    X.append(img)
    Y.append(mask)

X = np.array(X)
Y = np.array(Y)

model = unet()
model.compile(
    optimizer=Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(X, Y, epochs=15, batch_size=2)
model.save("models/unet_brain_mri.h5")

print("✅ U-Net model trained and saved!")
