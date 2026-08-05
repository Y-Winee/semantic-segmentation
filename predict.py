import torch
import numpy as np

from config import PREPROCESS
from model import DEVICE

def predict(model, image_np):

    transformed = PREPROCESS(image=image_np)

    tensor = transformed["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        logits = model(tensor)

        mask = torch.argmax(
            logits,
            dim=1
        ).squeeze().cpu().numpy()

    return mask