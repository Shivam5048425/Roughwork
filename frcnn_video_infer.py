import torch
import cv2
from torchvision import transforms
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import time
import os

# -----------------------------
# Configuration
# -----------------------------
VIDEO_PATH = "s311.mp4"  # Input video
OUTPUT_PATH = "output_faster_rcnn.mp4"  # Output annotated video
PTH_PATH = "frcnn_stage_High.pth"  # Your trained Faster R-CNN weights
NUM_CLASSES = 3  # Set to the number of classes in your trained model (including background)
CONF_THRESHOLD = 0.25  # Minimum score to show a detection

# -----------------------------
# Load model
# -----------------------------
# Initialize model architecture
model = fasterrcnn_resnet50_fpn(num_classes=NUM_CLASSES)

# Load weights
state_dict = torch.load(PTH_PATH)
model.load_state_dict(state_dict)
model.eval()

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -----------------------------
# Open video
# -----------------------------
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Video file {VIDEO_PATH} not found!")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video {VIDEO_PATH}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps_input,
    (width, height)
)

# -----------------------------
# Preprocessing transform
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
])

# -----------------------------
# Inference loop
# -----------------------------
frame_count = 0
total_time = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # Convert BGR to RGB and transform
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)

    # Draw boxes
    boxes = outputs[0]['boxes'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        if score >= CONF_THRESHOLD:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}:{score:.2f}', (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Write annotated frame
    out.write(frame)

    # FPS measurement
    frame_time = time.time() - start_time
    total_time += frame_time
    frame_count += 1

# -----------------------------
# Release resources
# -----------------------------
cap.release()
out.release()

# Compute average FPS safely
if total_time > 0 and frame_count > 0:
    avg_fps = frame_count / total_time
    print(frame_count)
    print(total_time)
    print(f"⚡ Average FPS: {avg_fps:.2f}")
else:
    print("No frames were processed. Check the video path or format.")

print(f"✅ Inference complete! Video saved as: {OUTPUT_PATH}")

