import cv2
import json
import torch
import shutil
import numpy as np
from pathlib import Path

import albumentations as A
from albumentations import ToTensorV2

import segmentation_models_pytorch as smp

INPUT_DIR = Path("./ds/v2-without-annotated")
OUTPUT_DIR = Path("./v2-predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_WEIGHTS = "./best_deeplabv3_hitl.pth"
NO_CLASSES = 4

CATEGORIES_NAMES = {
    1 : "road",
    2 : "road-marking",
    3 : "vehicle"
}

# --- COCO JSON Initialization ---
coco_format = {
    "info": {"description": "HITL Predictions"},
    "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
    "images": [],
    "annotations": [],
    "categories": [{"id": id, "name": name, "supercategory": "none"}for id, name in CATEGORIES_NAMES.items()]
}

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
base_model = smp.DeepLabV3(encoder_name="resnet34", classes=NO_CLASSES, in_channels=3)
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
base_model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
base_model.to(device)
base_model.eval()

image_paths = INPUT_DIR.glob("*.jpg")
annotation_id = 1

with torch.no_grad():
    for img_id, img_path in enumerate(image_paths, start=1):
        filename = img_path.name
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        original_h, original_w = img_bgr.shape[:2]
        coco_format['images'].append({
            "id": img_id,
            "license": 1,
            "file_name": filename,
            "height": original_h,
            "width": original_w
        })

        shutil.copy(img_path, OUTPUT_DIR/filename)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor_img = transform(image=img_rgb)['image'].unsqueeze(0).to(device)
        logits = base_model(tensor_img) # ** 
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        resized_pred = cv2.resize(pred.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST)

        for class_id in CATEGORIES_NAMES.keys(): # **
            binary_mask = (resized_pred == class_id).astype(np.uint8)

            if not np.any(binary_mask):
                continue

            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 50:
                    continue

                epsilon = 0.002 * cv2.arcLength(contour, True)
                contour = cv2.approxPolyDP(contour, epsilon, True)

                polygon = contour.flatten().tolist()

                if len(polygon) < 6:
                    continue

                x, y, w, h = cv2.boundingRect(contour)

                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": class_id,
                    "segmentation": [polygon],
                    "bbox": [x, y, w, h],
                    "area": float(area),
                    "iscrowd": 0
                })
                annotation_id += 1
coco_out_path = OUTPUT_DIR / "_annotations.coco.json"
with open(coco_out_path, "w") as f:
    json.dump(coco_format, f, indent=4)

print(f"Predictions saved to {OUTPUT_DIR}")