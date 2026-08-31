import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================
# PREPROCESSING
# ============================================================

transform = A.Compose([
    A.Resize(512, 512),
    A.CenterCrop(512, 512),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])


# ============================================================
# PREDICT
# ============================================================

def predict(model, image):

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = next(model.parameters()).device

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    transformed = transform(
        image=image
    )

    input_tensor = transformed["image"]

    # Add batch dimension
    input_tensor = input_tensor.unsqueeze(0)

    # Move to GPU / CPU
    input_tensor = input_tensor.to(
        device,
        non_blocking=True
    )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    model.eval()

    with torch.inference_mode():

        output = model(input_tensor)

        # segmentation_models_pytorch normally
        # returns a tensor directly
        logits = output

        mask = torch.argmax(
            logits,
            dim=1
        )

    # --------------------------------------------------------
    # TO NUMPY
    # --------------------------------------------------------

    mask = mask.squeeze(0).cpu().numpy()

    return mask