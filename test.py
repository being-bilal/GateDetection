import os
import csv
import cv2
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt

MODELS_DIR = "models"
DATA_YAML = "data.yaml"
TEST_IMAGES = "test_images"
OUTPUT_DIR = "eval_outputs"
CONF = 0.5
IOU = 0.5
IMG_SIZE = 640
DEVICE = "cuda"   

os.makedirs(OUTPUT_DIR, exist_ok=True)
VIS_DIR = os.path.join(OUTPUT_DIR, "visuals")
os.makedirs(VIS_DIR, exist_ok=True)


results_table = []

model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pt")]

print(f"Evaluating {len(model_files)} models...")

for model_file in model_files:
    model_path = os.path.join(MODELS_DIR, model_file)
    model_name = model_file.replace(".pt", "")
    print(f"\nRunning evaluation for: {model_name}")

    model = YOLO(model_path)

    metrics = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        conf=CONF,
        iou=IOU,
        device=DEVICE,
        plots=True,
        save_json=True,
        project=OUTPUT_DIR,
        name=model_name
    )

    results_table.append({
        "model": model_name,
        "mAP@0.5": metrics.box.map50,
        "mAP@0.5:0.95": metrics.box.map,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr
    })

    # ---------- VISUAL DIFFS ----------
    model_vis_dir = os.path.join(VIS_DIR, model_name)
    os.makedirs(model_vis_dir, exist_ok=True)

    for img_name in os.listdir(TEST_IMAGES):
        img_path = os.path.join(TEST_IMAGES, img_name)
        img = cv2.imread(img_path)

        preds = model(
            img,
            conf=CONF,
            imgsz=IMG_SIZE,
            device=DEVICE
        )

        annotated = preds[0].plot()
        save_path = os.path.join(model_vis_dir, img_name)
        cv2.imwrite(save_path, annotated)

# ---------------- SAVE CSV ---------------- #
df = pd.DataFrame(results_table)
csv_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
df.to_csv(csv_path, index=False)

print("\nEvaluation complete.")
print(df)
