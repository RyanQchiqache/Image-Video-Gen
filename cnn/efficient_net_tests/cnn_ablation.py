import os
from datetime import datetime
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter 

from tqdm import tqdm
from loguru import logger

import torchvision
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

import matplotlib.pyplot as plt

# -----------------------------
# small log_dir generation
#-------------------------------
PATH = "/home/ryqc/projects/python_projects/Image-Video-Gen/cnn/experiments_cnnAblation"
run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join(PATH, "runs",f"effb0_cifar10_{run_name}" )
writer = SummaryWriter(log_dir=log_dir)


# -----------------------------
# Config
# -----------------------------
LR = 4e-4
EPOCHS = 70
BATCH_SIZE = 64
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 10
RESIZE = (224, 224)  # EfficientNet-B0 expects 224x224
FREEZE_BACKBONE = False  

# -----------------------------
# Transforms (Correct order + correct normalization)
# -----------------------------
weights = EfficientNet_B0_Weights.DEFAULT

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.Resize(RESIZE),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=imagenet_mean, std=imagenet_std),
])

test_transform = T.Compose([
    T.Resize(RESIZE),
    T.ToTensor(),
    T.Normalize(mean=imagenet_mean, std=imagenet_std),
])


# -----------------------------
# Datasets / Loaders
# -----------------------------
train_dataset: CIFAR10 = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
test_dataset: CIFAR10 = CIFAR10(root="./data", train=False, download=True, transform=test_transform)

classes = train_dataset.classes
logger.info(f"Classes: {classes}")

train_loader: DataLoader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=NUM_WORKERS,
)

test_loader: DataLoader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,  # IMPORTANT: do not shuffle validation/test
    pin_memory=True,
    num_workers=NUM_WORKERS,
)


# -----------------------------
# Optional: visualize samples (denormalize for correct display)
# -----------------------------

images_per_class: Dict[int, List[Any]] = {label: [] for label in range(NUM_CLASSES)}
for img, label in train_dataset:
    if len(images_per_class[label]) < 10:
        images_per_class[label].append(img)

fig, axes = plt.subplots(NUM_CLASSES, 10, figsize=(20, 20))
for c in range(NUM_CLASSES):
    for i in range(10):
        img = images_per_class[c][i].permute(1,2,0).numpy()
        ax = axes[c][i]
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(classes[c])
plt.tight_layout()
# plt.show()


# -----------------------------
# Model (Correct: weights + classifier head replacement)
# -----------------------------
model = efficientnet_b0(weights=weights).to(DEVICE)

# Replace classifier head to output 10 classes
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES).to(DEVICE)

# Optionally freeze backbone for linear probing
if FREEZE_BACKBONE:
    for p in model.features.parameters():
        p.requires_grad = False
    logger.info("Backbone frozen: training classifier head only.")
else:
    logger.info("Full fine-tuning: training all parameters.")

params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
logger.info(f"EfficientNet-B0 trainable parameters: {params_m:.2f}M")


# -----------------------------
# Criterion / Optim / AMP
# -----------------------------
criterion = nn.CrossEntropyLoss()
optim = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

# -----------------------------
# Metric
# -----------------------------
def num_correct(logits: torch.Tensor, labels: torch.Tensor) -> int:
    preds = logits.argmax(dim=1)
    return (preds == labels).sum().item()


# -----------------------------
# Validation
# -----------------------------
@torch.no_grad()
def validation(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    for img, label in tqdm(loader, desc="Validation", leave=False):
        img, label = img.to(device), label.to(device)
        output = model(img)

        correct += num_correct(output, label)
        total += label.size(0)

    return 100.0 * correct / max(total, 1)


# -----------------------------
# Training
# -----------------------------
def training(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    criterion: nn.Module,
    device: torch.device,
    optim: torch.optim.Optimizer,
):
    model.train()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for img, label in tqdm(train_loader, desc=f"Training {epoch+1}/{epochs}"):
            img, label = img.to(device), label.to(device)

            optim.zero_grad(set_to_none=True)

            with autocast(enabled=(device.type == "cuda")):
                output = model(img)
                loss = criterion(output, label)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            bs = label.size(0)
            total_loss += loss.item() * bs
            total_correct += num_correct(output, label)
            total_samples += bs

        epoch_loss = total_loss / max(total_samples, 1)
        epoch_acc = 100.0 * total_correct / max(total_samples, 1)

        val_acc = validation(model, test_loader, device)

        logger.info(
            f"Epoch: {epoch+1}/{epochs} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Train Acc: {epoch_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}%"
        )

        writer.add_scalar("Loss", epoch_loss)
        writer.add_scalar("Train Acc", epoch_acc)
        writer.add_scalar("Val Acc", val_acc)

        if (epoch+1) % 10 == 0:
            os.makedirs(PATH, exist_ok=True)
            ckpt_path = os.path.join(PATH, f"efficient_net_epoch_{epoch}.pt")
            torch.save(
                {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer":optim.state_dict(),
                "scaler":scaler.state_dict()
                },
                ckpt_path,


            )

    


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    training(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=EPOCHS,
        criterion=criterion,
        device=DEVICE,
        optim=optim,
    )




