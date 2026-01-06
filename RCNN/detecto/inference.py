import os
import cv2
import random

# ===== CONFIG =====
DATASET_DIR = "/Users/mohammadbilal/Documents/Projects/GateDetection/Dataset"
RESULTS_DIR = "/Users/mohammadbilal/Documents/Projects/GateDetection/Dataset/visualizations"
SPLITS = ["train", "valid", "test"]
NUM_SAMPLES = 100
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# YOLO classes mapping
CLASS_NAMES = {
    0: "Gate",  # merged 6+12
    1: "Flair"
}
COLORS = {
    0: (0, 255, 0),   # green
    1: (255, 0, 0)    # blue
}
# ==================

for split in SPLITS:
    img_dir = os.path.join(DATASET_DIR, split, "images")
    lbl_dir = os.path.join(DATASET_DIR, split, "labels")
    out_dir = os.path.join(RESULTS_DIR, split)
    os.makedirs(out_dir, exist_ok=True)

    # List all images
    images = [f for f in os.listdir(img_dir) if f.lower().endswith(IMAGE_EXTS)]
    random.shuffle(images)
    images = images[:NUM_SAMPLES]

    for img_name in images:
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Failed to read {img_path}")
            continue

        h, w, _ = img.shape

        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        print(f"⚠️ Skipping invalid line in {lbl_path}: {line}")
                        continue

                    # Take only first 5 values
                    cls, x, y, bw, bh = map(float, parts[:5])
                    cls = int(cls)

                    # YOLO → pixel coordinates
                    x1 = int((x - bw / 2) * w)
                    y1 = int((y - bh / 2) * h)
                    x2 = int((x + bw / 2) * w)
                    y2 = int((y + bh / 2) * h)

                    color = COLORS.get(cls, (255, 255, 255))
                    label = CLASS_NAMES.get(cls, f"class_{cls}")

                    # Draw box and label
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        img,
                        label,
                        (x1, max(y1 - 7, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

        out_path = os.path.join(out_dir, img_name)
        cv2.imwrite(out_path, img)

    print(f"✅ Saved {len(images)} visualized images for {split} split in {out_dir}")
