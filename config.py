import albumentations as A
from albumentations.pytorch import ToTensorV2

NO_CLASSES = 4

COLORS = {
    0: [0, 0, 0],
    1: [255, 0, 0],
    2: [0, 255, 0],
    3: [0, 0, 255],
}

PREPROCESS = A.Compose([
    A.Resize(512, 512),
    A.CenterCrop(512, 512),
    A.Normalize(
        mean=(0.485,0.456,0.406),
        std=(0.229,0.224,0.225)
    ),
    ToTensorV2()
])