import json
import shutil
from pathlib import Path

def extract_images(input_dir, output_dir, json_file):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(json_file, "r") as f:
        coco_data = json.load(f)

    manual_images = set(
        img['extra']['name']
        for img in coco_data['images']
    )

    unlabelled_images = [
       img for img in input_dir.glob("*.jpg")
       if img.name not in manual_images
    ]

    for img in unlabelled_images:
        shutil.copy(img, output_dir/img.name)

    print(f"Copied {len(unlabelled_images)} images.")

if __name__ == "__main__":
    extract_images(input_dir="./ds/semantic_dataset", output_dir="./ds/v2-without-110", json_file="./ds/_annotations.coco.json")