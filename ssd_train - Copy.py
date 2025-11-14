

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

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm


import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead


import albumentations as A
from albumentations.pytorch import ToTensorV2
from usaugment.albumentations import DepthAttenuation, GaussianShadow, HazeArtifact, SpeckleReduction



# ---------- Visualization (Original + Labeled Augmentations) ----------
def visualize_multiple_augmentations(dataset, num_samples=5):
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
        DepthAttenuation(p=1.0, attenuation_rate=1.0),
        ToTensorV2()
  ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
       additional_targets={"scan_mask": "mask"})),

    (" GaussianShadow", A.Compose([
        GaussianShadow(p=1.0),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
       additional_targets={"scan_mask": "mask"})),

    (" HazeArtifact", A.Compose([
         HazeArtifact(p=1.0),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
     additional_targets={"scan_mask": "mask"})),

    #("SpeckleReduction", A.Compose([
    #    SpeckleReduction(p=1.0),
    #    ToTensorV2()
    #], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    #   additional_targets={"scan_mask": "mask"})), 

        #  ("LongestMaxSize", A.Compose([
        #    A.LongestMaxSize(max_size=300),
        #     ToTensorV2()
        #                                  ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        #                                            additional_targets={"scan_mask": "mask"})),

        

        # ("PadIfNeeded", A.Compose([
        #    A.PadIfNeeded(min_height=300, min_width=300),
        #     ToTensorV2()
        #                                  ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        #                                            additional_targets={"scan_mask": "mask"})),

            (" Resize", A.Compose([
             A.Resize(height=300, width=300),
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
        plt.savefig("Visualization of different Augmentation settings")
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
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        
        #Select all the rows for this image
        records = self.data[self.data["image_name"] == img_name]
        boxes = records[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=np.float32)
        labels = records["class_id"].to_numpy(dtype=np.int64)

        if self.use_albumentations and self.transform is not None:
            dummy_mask = np.ones(image.shape[:2], dtype=np.uint8)
            transformed = self.transform(
                                            image=image, 
                                            scan_mask=dummy_mask, 
                                            bboxes=boxes, 
                                            labels=labels)
            
            image = transformed["image"].float()
            boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            labels = torch.tensor(transformed["labels"], dtype=torch.int64)
        else:
            #Convert to tensors without augmentation
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

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
        
# ----------------------------- Training -----------------------------
def train_ssd(model, train_loader, val_loader, optimizer, device, epochs=2000, patience=10):
    best_val_loss = float("inf")
    patience_counter = 0

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"[Epoch {epoch+1}] Training Loss: {avg_train_loss:.4f}")

    
        # ---------------- Validation ----------------
        model.eval()
        val_loss = 0.0
        confidence_threshold = 0.5
        filtered_outputs = []
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                
                model.train()
                try:
                    loss_dict = model(images, targets)
                    val_loss += sum(loss_dict.values()).item()
                except Exception:
                    val_loss += 0.0
                    
                                
                #Filter predictions
                model.eval()
                outputs = model(images)
                for out in outputs:
                           keep = out["scores"] > confidence_threshold
                           filtered_outputs.append({
                                   "boxes": out["boxes"][keep],
                                   "scores": out["scores"][keep],
                                   "labels": out["labels"][keep]
                                    })               
                       

            

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"[Epoch {epoch+1}] Validation Loss: {avg_val_loss:.4f}")

        # ---------------- Validation ----------------
        with torch.no_grad():
            metrics = evaluate_ssd(model, val_loader, device)  # returns dict
            mAP = metrics.get("map", 0.0)
        print(f"[Epoch {epoch+1}] mAP: {mAP:.4f}")

        # ---------------- Early Stopping ----------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "outputs/best_model.pth")
            print("✅ Model saved (best validation loss)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # ---------------- Return losses ----------------
    loss_df = pd.DataFrame({
        "epoch": range(1, len(train_losses) + 1),
        "train_loss": train_losses,
        "val_loss": val_losses
    })

    return model, loss_df






# ----------------------------- Evaluation -----------------------------
test_csv =

 


def evaluate_ssd(model, dataloader, device, confidence_threshold=0.5):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", max_detection_thresholds=[1000, 1000, 1000])
    metric.warn_on_many_detection = False
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            filtered_outputs = []
            for out in outputs:
                keep = out["scores"] > confidence_threshold
                filtered_outputs.append({
                    "boxes": out["boxes"][keep].cpu(),
                    "scores": out["scores"][keep].cpu(),
                    "labels": out["labels"][keep].cpu()
                })

            targets_cpu = [{k:v.cpu() for k, v in t.items()} for t in targets]        
            metric.update (filtered_outputs, targets_cpu)
    results = metric.compute()
    return results




# ----------------------------- Run -----------------------------
if __name__ == "__main__":
    train_csv = "labels/frcnn_labels_train.csv"
    train_dir = "images/train"
    val_csv = "labels/frcnn_labels_val.csv"
    val_dir = "images/val"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    IMAGE_MEAN = (0.485, 0.456, 0.406)
    IMAGE_STD = (0.229, 0.224, 0.225)
    
    # ---- USAugment + Albumentations ----
    train_transform = A.Compose([
        A.Resize(height=300, width=300),
         A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        
    
        
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=5, p=0.5),
        DepthAttenuation(p=1.0, attenuation_rate=1.0),
        
        GaussianShadow(p=1.0),
        HazeArtifact(p=0.0),
        #SpeckleReduction(p=1.0),
        
        ToTensorV2()
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
       additional_targets={"scan_mask": "mask"})

    val_transform = A.Compose([
        A.Resize(height=300, width=300),
        
        A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
                                    ToTensorV2()
                                        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
                                            additional_targets={"scan_mask": "mask"})

    # ---- Datasets ----
    train_dataset = USDataset(train_csv, train_dir, transform=train_transform)
    val_dataset = USDataset(val_csv, val_dir, transform=val_transform)

    print("🔍 Visualizing labeled augmentations...")
    visualize_multiple_augmentations(train_dataset, num_samples=5)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    num_classes = 3
    weights = SSD300_VGG16_Weights.COCO_V1
    model = ssd300_vgg16(weights=weights)
    
    num_anchors = model.anchor_generator.num_anchors_per_location()
    
    in_channels= [512, 1024, 512, 256, 256, 256]

    model.head.classification_head = SSDClassificationHead(
        in_channels = in_channels,
        num_anchors = num_anchors,
        num_classes = num_classes)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    

    model, loss_df = train_ssd(model, train_loader, val_loader, optimizer, device)
    # --- Save outputs AFTER training ---
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/best_ssd.pth")
    loss_df.to_csv("outputs/ssd_losses.csv", index=False)
    print("Model and training losses saved to 'outputs/'")

    loss_df.to_csv("ssd_losses.csv", index=False)
    print("Training and validation losses saved to ssd_losses.csv")
    
    #---Evaluation---------
    results = evaluate_ssd(model, val_loader, device)
    # Check what map_per_class contains
    print("results['map_per_class']:", results["map_per_class"])
    #---convert per-class mAP tensor to dict
    per_class_map = {i: v.item() for i, v in enumerate(results["map_per_class"])}
    mAP_50_95 = results["map"].item() 
    precision = results.get("precision", 0.0) 
    recall = results.get("recall", 0.0)
   
    metrics = pd.DataFrame([{
        "precision": precision,
        "recall": recall,
        "mAP50-95_all": mAP_50_95,
        "mAP50-95_needle": per_class_map.get(1, 0.0),
        "mAP50-95_lesion": per_class_map.get(2, 0.0)
    }])
    metrics.to_csv("ssd_metrics.csv", index=False)
    print("✅ Final metrics saved to ssd_metrics.csv")

    #------------Visualise Predictions
    model.eval()
    print("\n Visualizing predictions vs ground truth")

    #Pick one sample from the validation dataset
    img, target = val_dataset[15]
    img_tensor = img.to(device)

    with torch.no_grad():
        pred = model([img_tensor])[0]

    #Convert image back to [0-255] uint8 for drawing 
    img_for_draw = (img_tensor * 255).to(torch.uint8).cpu()

    #Draw predicted boxes (red)
    img_pred = torchvision.utils.draw_bounding_boxes(
        img_for_draw,
        boxes=pred["boxes"].cpu(),
        labels=[str(l.item()) for l in pred["labels"]],
        colors='red',
        width=2
    )
    
    #Draw ground truth boxes (green)
    img_gt = torchvision.utils.draw_bounding_boxes(
        img_for_draw,
        boxes=target["boxes"],
        labels=[str(l.item()) for l in target["labels"]],
        colors="green",
        width=2
    )
 
    #Plot both side by side 
    fig, axs = plt.subplots(1, 2, figsize=(12,6))
    axs[0].imshow(img_gt.permute(1,2,0))
    axs[0].set_title("Ground Truth (Green)")
    axs[0].axis("off")

    axs[1].imshow(img_pred.permute(1,2,0))
    axs[1].set_title("Predictions(Red)")
    axs[1].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/predictions_vs_groundtruth.png")
    plt.show()