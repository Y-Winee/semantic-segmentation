import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class SegmentationDataset(Dataset):
    def __init__(self, input_dir, annotation_file, transform=None):
        super().__init__()

        self.input_dir = input_dir
        self.annotation_file = annotation_file
        self.transform = transform

        self.coco = COCO(annotation_file)
        self.img_ids = self.coco.getImgIds()

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        # Load image
        img_path = os.path.join(
            self.input_dir,
            img_info["file_name"]
        )

        image = np.array(
            Image.open(img_path).convert("RGB")
        )

        # 0 = background
        # 1 = road
        # 2 = road-marking
        # 3 = vehicle
        mask = np.zeros(
            (img_info["height"], img_info["width"]),
            dtype=np.int64
        )

        # Get annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        for ann in anns:
            category_id = ann["category_id"]

            # Ignore category 0 ("okinawa-dashcam-mask")
            # if it appears in the COCO annotations.
            if category_id not in [1, 2, 3]:
                continue

            ann_mask = self.coco.annToMask(ann)

            mask[ann_mask == 1] = category_id

        # Apply augmentation / preprocessing
        if self.transform:
            transformed = self.transform(
                image=image,
                mask=mask
            )

            image = transformed["image"]
            mask = transformed["mask"].long()

        else:
            image = (
                torch.from_numpy(image)
                .permute(2, 0, 1)
                .float()
                / 255.0
            )

            mask = torch.from_numpy(mask).long()

        return image, mask