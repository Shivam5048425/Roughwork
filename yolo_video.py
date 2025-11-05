# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 16:14:32 2025

@author: barbate
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/k_fold_2_run_2/weights/best.pt")

# Path to your input video
video_path = "s311.mp4"  # <-- Change this

# Run inference on the video
results = model.predict(
    source=video_path,
    save=True,         # Save annotated video
    save_txt=True,    # Optionally save detections in YOLO txt format
    conf=0.50,         # Confidence threshold
    show=False,        # Set to True to display live during inference
    device=0,
        
)

print("Inference complete!")
print("Results saved in:", results[0].save_dir)