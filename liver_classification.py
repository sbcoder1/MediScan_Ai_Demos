import cv2, os, shutil, numpy as np

IMG_DIR = "dataset/liver_train/images"
MASK_DIR = "dataset/liver_train/masks"

OUT_NORMAL = "dataset/liver_classification/Normal"
OUT_TUMOR = "dataset/liver_classification/Liver_Tumor"

os.makedirs(OUT_NORMAL, exist_ok=True)
os.makedirs(OUT_TUMOR, exist_ok=True)

for f in os.listdir(IMG_DIR):
    img = os.path.join(IMG_DIR, f)
    mask = os.path.join(MASK_DIR, f)

    if not os.path.exists(mask):
        continue

    m = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)

    if np.sum(m) == 0:
        shutil.copy(img, os.path.join(OUT_NORMAL, f))
    else:
        shutil.copy(img, os.path.join(OUT_TUMOR, f))

print("✔ liver_classification dataset ready")
