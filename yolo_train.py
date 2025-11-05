# -*- coding: utf-8 -*-
"""
Train YOLOv8 with the same USAugment + Albumentations augmentations
used in Faster R-CNN (DepthAttenuation, GaussianShadow, HazeArtifact, SpeckleReduction).

Applies augmentations via an on_preprocess_batch callback for consistency.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
from usaugment.albumentations import DepthAttenuation, GaussianShadow, HazeArtifact, SpeckleReduction
import matplotlib.pyplot as plt


# ---------------- Albumentations Transform ----------------
alb_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=5, p=0.5),
        DepthAttenuation(p=1.0, attenuation_rate=1.0),
        
        GaussianShadow(p=1.0),
        HazeArtifact(p=1.0),
        #SpeckleReduction(p=0.5),
        ToTensorV2()
    ],
    additional_targets={"scan_mask": "mask"}
)


# ---------------- Apply Transform on YOLO Batch ----------------
def custom_albu_augment(batch):
    """
    Apply the same Albumentations + USAugment pipeline on each YOLO batch.
    Input batch: dict with "img" tensor of shape [N,3,H,W] in [0,1]
    """
    imgs = batch["img"].cpu().numpy().transpose(0, 2, 3, 1)  # [N,H,W,C]
    out_imgs = []

    for im in imgs:
        # Convert to uint8 image [0–255]
        im_uint8 = (im * 255).astype(np.uint8)

        # Apply transformations
        transformed = alb_transform(image=im_uint8, scan_mask=np.ones(im_uint8.shape[:2], dtype=np.uint8))
        aug_img = transformed["image"].permute(1, 2, 0).numpy()  # tensor → numpy (HWC)
        out_imgs.append(aug_img)

    # Stack back to YOLO format [N,3,H,W]
    batch["img"] = np.stack(out_imgs).transpose(0, 3, 1, 2).astype(np.float32)
    batch["img"] /= 255.0
    return batch


# ---------------- Visualization (Optional) ----------------
def visualize_augmentations(num_samples=3):
    """
    Visualize a few sample augmentations to verify effects.
    """
    import cv2
    for i in range(num_samples):
        img = cv2.imread("sample_ultrasound.png")[:, :, ::-1]  # Replace with one of your dataset images
        transformed = alb_transform(image=img, scan_mask=np.ones(img.shape[:2], dtype=np.uint8))
        aug_img = transformed["image"].permute(1, 2, 0).numpy()

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(np.clip(aug_img, 0, 1))
        plt.title("Augmented")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


# ---------------- YOLOv8 Training ----------------
if __name__ == "__main__":
   
    # Load base YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Register callback that applies Albumentations transforms before each batch
    #model.add_callback("on_preprocess_batch", lambda trainer, batch: custom_albu_augment(batch))

    # Start training (same params as in Faster R-CNN for fair comparison)
    results = model.train(
        data="coco8.yaml",  # Replace with your dataset YAML
        epochs=2000,
        workers=1,
        batch=8,
        lr0=0.000025,
        optimizer="SGD",
        augment=False,   # disable YOLO internal augmentations
        mosaic=0.0,
        #degrees=0.5,
        #fliplr=0.5
    )

   

