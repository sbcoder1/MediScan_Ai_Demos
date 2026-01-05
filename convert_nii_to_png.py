import nibabel as nib
import numpy as np
import cv2
import os

# ---------------- PATHS ----------------
INPUT_FOLDER = "BraTS20_Training_002"
OUTPUT_IMAGE_FOLDER = "dataset/images/flair"
OUTPUT_MASK_FOLDER = "dataset/masks"

os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_MASK_FOLDER, exist_ok=True)

# ---------------- LOAD FILES ----------------
flair_path = os.path.join(INPUT_FOLDER, "BraTS20_Training_002_flair.nii")
mask_path = os.path.join(INPUT_FOLDER, "BraTS20_Training_002_seg.nii")

flair = nib.load(flair_path).get_fdata()
mask = nib.load(mask_path).get_fdata()

print("Loaded MRI shape:", flair.shape)

# ---------------- CONVERT ----------------
for i in range(flair.shape[2]):

    img_slice = flair[:, :, i]
    mask_slice = mask[:, :, i]

    # skip empty tumor slices
    if np.max(mask_slice) == 0:
        continue

    # normalize image
    img_slice = cv2.normalize(img_slice, None, 0, 255, cv2.NORM_MINMAX)
    img_slice = img_slice.astype(np.uint8)

    # binary mask
    mask_slice = (mask_slice > 0).astype(np.uint8) * 255

    cv2.imwrite(f"{OUTPUT_IMAGE_FOLDER}/flair_{i}.png", img_slice)
    cv2.imwrite(f"{OUTPUT_MASK_FOLDER}/mask_{i}.png", mask_slice)

print("✅ Done! PNG images created.")
