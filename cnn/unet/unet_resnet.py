import os
import cv2
import rasterio 
import torch
import segmentation_models_pytorch as smp 
import numpy as np

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.tensorboard import SummaryWriter
import torchvision.transforms as T
from torchmetrics.classification import MulticlassJaccardIndex

from torch.optim import Adam
from torch.utils.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from loguru import logger



##########################################################################
# CONFIG
##########################################################################

EPOCHS=100
LR=4e-4
BATCH_SIZE=16
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE=512
PATH_DATASET=""
NUM_CLASSES=20
IN_CHANNELS = 3
scaler = GradScaler()
writer = SummaryWriter()
PATH = "/project/python_projects/Image-Video/cnn/experiments"


##########################################################################
# DATA PREPROCESSING
##########################################################################


##########################################################################
# TRANSOFORMS
##########################################################################
train_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(
                mean = [0.5 for _ in range(IN_CHANNELS)]
                std = [0.5 for _ in range(IN_CHANNELS)],
            )
    ])

val_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5 for _ in range(IN_CHANNELS)])

    ])



##########################################################################
# MODEL, OPTIM, LOSS 
##########################################################################
model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES
).to(DEVICE)

parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
logger.info(f"model is initialized with {parameters } parameters.")

criterion = CrossEntropyLoss()
optim = Adam(model.parameters(), lr=LR)
logger.info(f"optimizer and loss OK!!")

###########################################################################
# DATASET_CLASS
############################################################################

class _Dataset(Dataset):
    def __init__(self, images, masks, transform) -> None:
        super().__init__()

        self.images = images
        self.masks = masks
        self.transform = transform


    def __len__(self):
        return len(self.images)
    
    def load_image(self, path):
        if path.endswith(".tiff"):
            with rasterio.open(path) as f:
                img = f.read([1,2,3])
                img = np.transpose(img, (1,2,0))

        else:
            img = cv2.imread(path, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32)
        return img
            
    def load_mask(self, path):
        mask =  cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return mask.astype(np.uint64)

    def __getitem__(self, idx):
        image= self.load_image(self.images[idx])
        mask = self.load_mask(self.masks[idx])

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask


#############################################################################
# METRICS
##############################################################################

def iou(targets, labels, clss):
    """
        Jaccard Inde --> Intersection over Union
        args:
            - targets: logits
            - labels: masks
            - clss : classes
    """
    if targets.ndim() > labels.ndim():
        targets = targets.argmax(dim=1)

    target_c = (targets == clss)
    label_c = (labels == clss)

    intersection = (target_c & label_c).sum().float()
    union = (target_c | label_c).sum().float()

    return intersection / (union * 1e8)


def miou(targets, labels, num_classes):
    """
        Mean Intersection over Union over all classes
        args:
            - targets: logits
            - labels: masks
            - num_classes: number of classes that the model outputs
    """
    return torch.mean(torch.stack([iou(targets, labels, cls) for cls in range(num_classes)]))


def accuracy5(tagets, labels, k):
    """
        Accuracy over top5 classes
        args:
            - targets: logits
            - labels: masks
            - k: int with num_of_k

    """

    _, topk = targets.topk(k, dim=1)
    labels = labels.view(-1,1)
    return (topk == labels).any(dim=1).mean().float()




##################################################################################
# VALIDATION
##################################################################################

def _validation(model, val_loader, criterion, device, k):
    """
        Validation per epoch
        args:
            - model: models used 
            - val_loader: validation dataset using Dataloader
            - criterion: loss funciton (e.g CE, DE, BCE...)
    """

    model.eval()
    num_batches = 0
    val_loss = 0
    topk5 = 0
    miou = 0
    for images, labels in tqdm(val_loader, desc="Validation"):
        images = images.to(device)
        labels = lables.to(device)

        with torch.no_grad():
            output = model(images)
            loss = criterion(output, labels)

        
        val_loss += loss.item()
        topk5 += accuracy5(output, labels, k).item()
        num_batches += 1
        miou += miou(outputs, labels, NUM_CLASSES)


    return {
            "val_loss": val_loss / num_batches,
            "topk5": topk5 / num_batches,
            "miou": miou
            }

##################################################################################
# TRAINING 
##################################################################################

for epoch in range(EPOCHS):
    model.train()
    
    _loss = 0
    for images, labels in tqdm(train_loader, desc="Training"):
        image = images.to(DEVICE)
        label = labels.to(DEVICE)

        with autocast():

            output = model(images)
            loss  = criterion(output, labels)

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        _loss += loss.item()

    total_loss = _loss / len(train_loader)

    validation = _validation(model, val_loader, criterion, DEVICE, k=5)
    
    
    logger.info(f"Epoch: {epoch}  |"
                f"Total_loss: {total_loss} |"
                f"Validation_loss: {validation['val_loss']:.4f}  |"
                f"top5 accuracy {validation['topk5']:.4f} |"
                f"Validaiton mIoU: {valdation['miou']:.4f}"
                )

    writer.add_scalar("Loss/train", total_loss, epoch)
    writer.add_scalar("Loss/val", validation['val_loss'], epoch)
    writer_add_scaler("Accuracy/top5", validation["topk5"], epoch)
    writer.add_scaler("mIoU/Validaiton", validation["miou"])

    if epoch % 10 == 0:
        torch.save(
                { 
                    "epoch":epoch,
                    "model":model.state_dict(),
                    "optimizer": optim.state_dict(),
                 "scaler": scaler.state_dict(),
                    },
                f"{PATH}_unet_{epoch}.pt"

        )

writer.close()

    




