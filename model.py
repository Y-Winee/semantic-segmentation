import torch
import segmentation_models_pytorch as smp

from config import NO_CLASSES

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)

def load_model():

    model = smp.DeepLabV3(
        encoder_name="resnet34",
        classes=NO_CLASSES,
        in_channels=3
    )

    model.load_state_dict(
        torch.load(
            "best_deeplabv3_hitl.pth",
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    return model