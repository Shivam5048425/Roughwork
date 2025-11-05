import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from usaugment.albumentations import DepthAttenuation, GaussianShadow, HazeArtifact, SpeckleReduction

# ------------------ Configuration ------------------
dataset_csv = "labels/frcnn_labels_train.csv"
img_dir = "images/train"
save_dir = "preprocessed_images"
num_images = 60

os.makedirs(save_dir, exist_ok=True)

# ------------------ Preprocessing / Augmentations ------------------
preprocess_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=5, p=0.5),
    DepthAttenuation(p=1.0, attenuation_rate=1.0),
    GaussianShadow(p=1.0),
    HazeArtifact(p=1.0),
    SpeckleReduction(p=1.0),
    ToTensorV2()
], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))

# ------------------ Load CSV ------------------
data = pd.read_csv(dataset_csv)
bbox_cols = data.columns[2:6]
data[bbox_cols] = data[bbox_cols].apply(pd.to_numeric, errors="raise")

# ------------------ Process and Save ------------------
for i in range(min(num_images, len(data))):
    img_path = os.path.join(img_dir, data.iloc[i, 0])
    image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
    bbox = data.iloc[i, 2:6].to_numpy(dtype=np.float32)
    label = int(data.iloc[i, 1])
    
    # Dummy mask required by some augmentations
    dummy_mask = np.ones(image.shape[:2], dtype=np.uint8)
    transformed = preprocess_transform(image=image, scan_mask=dummy_mask, bboxes=[bbox], labels=[label])
    
    aug_img = transformed["image"].permute(1, 2, 0).numpy()
    aug_img = (np.clip(aug_img, 0, 1) * 255).astype(np.uint8)
    
    # Save image
    save_path = os.path.join(save_dir, f"{i+1}_{data.iloc[i,0]}")
    Image.fromarray(aug_img).save(save_path)

print(f"✅ Preprocessed {min(num_images, len(data))} images with SpeckleReduction saved to '{save_dir}'")
