import torch
from torch.utils.data import DataLoader, random_split

from coco_to_masks import SegmentationDataset

import segmentation_models_pytorch as smp

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchmetrics.segmentation import MeanIoU


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)

print(f"Using device: {device}")


# ============================================================
# TRANSFORMS
# ============================================================

transform = A.Compose([
    A.Resize(512, 512),

    A.HorizontalFlip(p=0.5),

    A.ShiftScaleRotate(
        shift_limit=0.005,
        scale_limit=0.10,
        rotate_limit=5,
        border_mode=0,
        p=0.5
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.3
    ),

    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),

    ToTensorV2()
])


# ============================================================
# DATASET
# ============================================================

INPUT_DIR = "./ds/v1-125"
ANNOTATION_FILE_DIR = "./ds/_annotations.coco.json"

total_ds = SegmentationDataset(
    input_dir=INPUT_DIR,
    annotation_file=ANNOTATION_FILE_DIR,
    transform=transform
)

print(f"Total images: {len(total_ds)}")


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

train_size = int(0.8 * len(total_ds))
val_size = len(total_ds) - train_size

train_ds, val_ds = random_split(
    total_ds,
    [train_size, val_size]
)

print(f"Training images: {len(train_ds)}")
print(f"Validation images: {len(val_ds)}")


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    dataset=val_ds,
    batch_size=32,
    shuffle=False
)


# ============================================================
# NUMBER OF CLASSES
# ============================================================

# 0 = background
# 1 = road
# 2 = road-marking
# 3 = vehicle

NUM_CLASSES = 4


# ============================================================
# MODEL
# ============================================================

base_model = smp.DeepLabV3(
    encoder_name="resnet34",
    classes=NUM_CLASSES,
    encoder_weights="imagenet",
    in_channels=3
).to(device)


# ============================================================
# LOSS
# ============================================================

dice = smp.losses.DiceLoss(
    mode="multiclass"
).to(device)

cross_entropy = torch.nn.CrossEntropyLoss().to(device)


def combined_loss(pred, target):
    dice_loss = dice(pred, target)
    ce_loss = cross_entropy(pred, target)

    return dice_loss + ce_loss


# ============================================================
# METRIC
# ============================================================

metric = MeanIoU(
    num_classes=NUM_CLASSES,
    per_class=False,
    input_format="index"
).to(device)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    metrics,
    n_epochs,
    patience=10,
    save_path="best_model.pth"
):

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(n_epochs):

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()
        metrics.reset()

        total_train_loss = 0.0

        for X_train, y_train in train_loader:

            X_train = X_train.to(device)
            y_train = y_train.to(device)

            optimizer.zero_grad()

            y_train_pred = model(X_train)

            loss = loss_fn(
                y_train_pred,
                y_train
            )

            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

            pred_indices = torch.argmax(
                y_train_pred,
                dim=1
            )

            metrics.update(
                pred_indices,
                y_train
            )

        avg_train_loss = (
            total_train_loss / len(train_loader)
        )

        train_metric = metrics.compute()


        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()
        metrics.reset()

        total_val_loss = 0.0

        with torch.no_grad():

            for X_val, y_val in val_loader:

                X_val = X_val.to(device)
                y_val = y_val.to(device)

                y_val_pred = model(X_val)

                loss = loss_fn(
                    y_val_pred,
                    y_val
                )

                total_val_loss += loss.item()

                pred_indices = torch.argmax(
                    y_val_pred,
                    dim=1
                )

                metrics.update(
                    pred_indices,
                    y_val
                )

        avg_val_loss = (
            total_val_loss / len(val_loader)
        )

        val_metric = metrics.compute()


        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if avg_val_loss < best_val_loss:

            best_val_loss = avg_val_loss
            epochs_no_improve = 0

            torch.save(
                model.state_dict(),
                save_path
            )

        else:
            epochs_no_improve += 1


        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print(
            f"Epoch [{epoch + 1}/{n_epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} - "
            f"Train IoU: {train_metric:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} - "
            f"Val IoU: {val_metric:.4f}"
        )


        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if epochs_no_improve >= patience:

            print(
                f"\nEarly stopping triggered "
                f"after {epoch + 1} epochs!"
            )

            break


# ============================================================
# STAGE 1
# FREEZE ENCODER
# ============================================================

for param in base_model.encoder.parameters():
    param.requires_grad = False


# ============================================================
# OPTIMIZER - STAGE 1
# ============================================================

optim = torch.optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        base_model.parameters()
    ),
    lr=1e-3,
    weight_decay=1e-4
)


# ============================================================
# TRAIN STAGE 1
# ============================================================

train(
    base_model,
    train_loader,
    val_loader,
    combined_loss,
    optim,
    metric,
    15,
    save_path="best_model.pth"
)


# ============================================================
# LOAD BEST STAGE 1 MODEL
# ============================================================

base_model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location=device
    )
)


# ============================================================
# STAGE 2
# UNFREEZE ENCODER
# ============================================================

for param in base_model.encoder.parameters():
    param.requires_grad = True


# ============================================================
# OPTIMIZER - STAGE 2
# ============================================================

optim_2 = torch.optim.AdamW(
    base_model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)


# ============================================================
# TRAIN STAGE 2
# ============================================================

train(
    base_model,
    train_loader,
    val_loader,
    combined_loss,
    optim_2,
    metric,
    85,
    save_path="best_deeplabv3_hitl.pth"
)


# ============================================================
# DONE
# ============================================================

print(
    "\nAll training complete! "
    "Best model saved as 'best_deeplabv3_hitl.pth'"
)