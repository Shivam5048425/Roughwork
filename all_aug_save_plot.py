# -*- coding: utf-8 -*-
"""
Train Faster R-CNN on a custom dataset with Early Stopping (YOLO-style).
Includes visualization of USAugment + Albumentations augmentations
(original + individually labeled augmentations).

Saves:
 - frcnn_losses.csv → training/validation losses per epoch
 - frcnn_metrics.csv → final evaluation metrics (mAP, precision, recall)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
import glob

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from torchvision import transforms
import torchvision.transforms.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2
from usaugment.albumentations import DepthAttenuation, GaussianShadow, HazeArtifact, SpeckleReduction

# ---------- Visualization (Original + Labeled Augmentations) ----------
def visualize_multiple_augmentations(dataset, num_samples=3):
    """
    Show the original image + individual augmentations with names.
    Each augmentation is applied separately to make its effect clear.
    """

    named_augs = [
        ("HorizontalFlip", A.Compose([
            A.HorizontalFlip(p=1.0),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
           additional_targets={"scan_mask": "mask"})),

        ("Rotation", A.Compose([
            A.Rotate(limit=5, p=1.0),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
           additional_targets={"scan_mask": "mask"})),

        ("DepthAttenuation", A.Compose([
            DepthAttenuation(p=1.0, attenuation_rate=0.3),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
           additional_targets={"scan_mask": "mask"})),
    ]

    for i in range(min(num_samples, len(dataset))):
        img_path = os.path.join(dataset.img_dir, dataset.data.iloc[i, 0])
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        bbox = dataset.data.iloc[i, 2:6].to_numpy(dtype=np.float32)
        label = int(dataset.data.iloc[i, 1])

        fig, axes = plt.subplots(1, len(named_augs) + 1, figsize=(4 * (len(named_augs) + 1), 4))
        fig.suptitle(f"Sample {i+1}: {os.path.basename(img_path)}", fontsize=13)

        # ---- Original ----
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis("off")
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                 linewidth=2, edgecolor="r", facecolor="none")
        axes[0].add_patch(rect)

        # ---- Individual augmentations ----
        for j, (name, aug) in enumerate(named_augs):
            dummy_mask = np.ones(image.shape[:2], dtype=np.uint8)
            transformed = aug(image=image, scan_mask=dummy_mask, bboxes=[bbox], labels=[label])
            aug_img = transformed["image"].permute(1, 2, 0).numpy()
            aug_img = np.clip(aug_img, 0, 1)
            aug_boxes = transformed["bboxes"]

            axes[j + 1].imshow(aug_img)
            axes[j + 1].set_title(name)
            axes[j + 1].axis("off")
            for box in aug_boxes:
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         linewidth=2, edgecolor="r", facecolor="none")
                axes[j + 1].add_patch(rect)

        plt.tight_layout()
        plt.show()



# ----------------------------- Dataset -----------------------------
class USDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, use_albumentations=True):
        self.data = pd.read_csv(csv_file)
        bbox_cols = self.data.columns[2:6]
        self.data[bbox_cols] = self.data[bbox_cols].apply(pd.to_numeric, errors="raise")
        self.img_dir = img_dir
        self.transform = transform
        self.use_albumentations = use_albumentations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        bbox = self.data.iloc[idx, 2:6].to_numpy(dtype=np.float32)
        label = int(self.data.iloc[idx, 1])

        if self.use_albumentations and self.transform is not None:
            dummy_mask = np.ones(image.shape[:2], dtype=np.uint8)
            transformed = self.transform(
                image=image, scan_mask=dummy_mask, bboxes=[bbox], labels=[label]
            )
            image = transformed["image"].float()
            
            boxes = torch.tensor(np.array(transformed["bboxes"]), dtype=torch.float32)
            labels = torch.tensor(transformed["labels"], dtype=torch.int64)
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
            boxes = torch.tensor([bbox], dtype=torch.float32)
            labels = torch.tensor([label], dtype=torch.int64)

        return image, {"boxes": boxes, "labels": labels}


def collate_fn(batch):
    return tuple(zip(*batch))




# ----------------------------- Early Stopping -----------------------------
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        improvement = (score - self.best_score) if self.mode == "max" else (self.best_score - score)
        if improvement > self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


# ----------------------------- Evaluation -----------------------------
def evaluate_frcnn(model, dataloader, num_classes=3, conf_thresh=0.5):
    model.eval()
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    iou_results = {cls: {thr: [] for thr in iou_thresholds} for cls in range(1, num_classes)}

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device).float() for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)

            for out in outputs:
                keep = out["scores"] >= conf_thresh
                out["boxes"] = out["boxes"][keep]
                out["labels"] = out["labels"][keep]

            for output, target in zip(outputs, targets):
                if len(output["boxes"]) == 0 or len(target["boxes"]) == 0:
                    continue
                ious = box_iou(output["boxes"].cpu(), target["boxes"].cpu())
                for t_idx, gt_label in enumerate(target["labels"].cpu().numpy()):
                    if len(ious) == 0:
                        continue
                    max_iou = ious[:, t_idx].max().item()
                    for thr in iou_thresholds:
                        iou_results[gt_label][thr].append(1 if max_iou >= thr else 0)

        tp_total, fp_total, fn_total = 0, 0, 0
        for output, target in zip(outputs, targets):
            if len(output["boxes"]) == 0 or len(target["boxes"]) == 0:
                continue
            ious = box_iou(output["boxes"].cpu(), target["boxes"].cpu())
            matched_gt = set()
            tp = 0
            for i, pred_box in enumerate(output["boxes"]):
                max_iou, gt_idx = ious[i].max(0)
                if max_iou >= 0.5 and gt_idx.item() not in matched_gt:
                    tp += 1
                    matched_gt.add(gt_idx.item())
            fp = len(output["boxes"]) - tp
            fn = len(target["boxes"]) - tp
            tp_total += tp
            fp_total += fp
            fn_total += fn

        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0

    per_class_map = {
        cls: np.mean([np.mean(vals) if len(vals) > 0 else 0.0 for vals in thr_dict.values()])
        for cls, thr_dict in iou_results.items()
    }
    mAP_50_95 = np.mean(list(per_class_map.values())) if per_class_map else 0.0
    return per_class_map, mAP_50_95, precision, recall


# ----------------------------- Training -----------------------------
def train_frcnn(model, train_loader, val_loader, optimizer, device, epochs=2000, patience=10):
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        print(f"\nEpoch [{epoch + 1}/{epochs}]")
        model.train()
        train_loss = 0.0

        for images, targets in train_loader:
            images = [img.to(device).float() for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Training Loss: {avg_train_loss:.4f}")

        val_loss = 0.0
        model.train()
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device).float() for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                val_loss += sum(loss for loss in loss_dict.values()).item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        model.eval()
        with torch.no_grad():
            per_class_map, mAP_50_95, precision, recall = evaluate_frcnn(model, val_loader)
        print(f"mAP[50:95]: {mAP_50_95:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    loss_df = pd.DataFrame({
        "epoch": range(1, len(train_losses) + 1),
        "train_loss": train_losses,
        "val_loss": val_losses
    })
    return model, loss_df

#-----------------Augmentation setting Configs----------------------------
CONFIGS = {
            "no_aug": {
                       "train_augs": [],
                        "run_name": "frcnn_no_aug" },

            "low_aug":{ "train_augs": [ A.HorizontalFlip(p=1.0),
                                        A.Rotate(limit=5, p=0.5)],
                        "run_name": "frcnn_low_setting" }
           }

def build_train_transform(augs):
    """Build Albumentations transform from config list"""
    return A.Compose(
        augs + [ToTensorV2()],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        additional_targets={"scan_mask": "mask"}
                            )

def run_training(config_key):
    cfg = CONFIGS[config_key]
    print(f"\n starting training for config: {config_key}")

    #-----Build augmentations-----------------
    train_transform = build_train_transform(cfg["train_augs"])
    val_transform = A.Compose([ToTensorV2()])

    #----Dataset paths------------------------
    train_csv = "labels/frcnn_labels_train.csv"
    train_dir = "images/train"
    val_csv = "labels/frcnn_labels_val.csv"
    val_dir = "images/val"

    #--------Dataset + loaders----------------------------------------------------------------------
    train_dataset = USDataset(train_csv, train_dir, transform=train_transform)
    img1, target1 = train_dataset[0]
    img2, target2 = train_dataset[8]
    print(img1, target1)
    print(target2)
    def show(imgs): 
            if not isinstance(imgs, list):
                imgs = [imgs]
            fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
            for i, img in enumerate(imgs):
                img = img.detach()
                img = F.to_pil_image(img)
                axs[0,i].imshow(np.asarray(img))
                axs[0,i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            plt.show()
    show([img1])
   
    val_dataset = USDataset(val_csv, val_dir, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    #---------Model---------------------------------
    num_classes = 3
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    #-----------------Train-----------------------------
    model, loss_df = train_frcnn(model, train_loader, val_loader, optimizer, device)

    #-----------------------Save results-------------------------------------
    run_name = cfg["run_name"]
    loss_df.to_csv(f"{run_name}_losses.csv", index=False)

    per_class_map, mAP_50_95, precision, recall = evaluate_frcnn(model, val_loader)
    metrics = pd.DataFrame([{

        "precision": precision,
        "recall": recall,
        "mAP50-95_all": mAP_50_95,
        "mAP50-95_needle": per_class_map.get(1, 0.0),
        "mAP50-95_lesion": per_class_map.get(2,0.0)
    }])
    metrics.to_csv(f"{run_name}_metrics.csv", index=False)
    print(f"finished {run_name} - results saved.")
    return model, loss_df, metrics
    
def plot_all_loss_curves():
    """reads all _losses.csv files and plots training vs validation losses 
        for each configuration on the same figure.
        """
    loss_files = glob.glob("*_losses.csv")
    if not loss_files:
        print("No loss CSV files found.")
        return

    plt.figure(figsize=(8,5))
    for f in loss_files:
        df = pd.read_csv(f)
        run_name = f.replace("_losses.csv", "")
        plt.plot(df["epoch"], df["train_loss"], label=f"{run_name} - train", linestyle="--")
        plt.plot(df["epoch"], df["val_loss"], label=f"{run_name} - val", linewidth=2)

    plt.title("Training vs Validation Loss (All Configs)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# ----------------------------- Run -----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_losses, all_metrics = [], []

    # Run both training phases sequentially---
    for cfg_key in ["no_aug", "low_aug"]:
        _, loss_df, metrics_df = run_training(cfg_key)
        all_losses.append(loss_df)
        all_metrics.append(metrics_df)
        
    # Concatenate all configs into single CSVs
    all_losses_df = pd.concat(all_losses, ignore_index=True)
    all_losses_df.to_csv("all_augment_losses.csv", index=False)

    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    all_metrics_df.to_csv("all_augment_metrics.csv", index=False)
    
    run_training(cfg_key)
    

    print("\n training phases complete.")

  
    
    

    #Plot all curves 
    plt_all_loss_curves()

