import torch
import torchvision
import torchvision.transforms as T

from tqdm import tqdm
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from datasets import load_dataset
from vision_transformers import ViT
from torch.cuda.amp import GradScaler, autocast


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################
# CONFIG
###############################################################
BATCHSIZE=64
EPOCHS=64
LR=3e-4
RESIZE=(320,320)

################################################################
# VIT CONFIG
##############################################################
IN_HEIGHT=320
IN_WIDTH=320
OUT_HEIGHT=320 // 16
OUT_WIDTH=320// 16
IN_CHANNELS= 3
DROPOUT=0.1
ATT_DROPOUT=0.1
EM_DIM=384
NUM_CLASSES=1000
MLP_SIZE=1536 # 4 x em_dim
NUM_LAYERS=12
NUM_HEADS=6 # (384 / 64)



################################################################
# DATASET
################################################################
# Login using e.g. `huggingface-cli login` to access this dataset
train_dataset = load_dataset("ILSVRC/imagenet-1k", split="train")
val_dataset = load_dataset("ILSVRC/imagenet-1k", split="validation")


##################################################################
# TRANSFORM & DATALOADER
##################################################################
train_transforms = T.Compose([
    T.Resize(RESIZE),
    T.RandomRotation(0.5),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])

val_transforms = T.Compose([
    T.Resize(RESIZE),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])

def train_hf_transform(example):
    image = example["image"]
    label = example["label"]
    image = train_transforms(image)
    return {"pixel_values": image, "labels": label}

def val_hf_transform(example):
    image = example["image"]
    label = example["label"]
    image = val_transforms(image)
    return {"pixel_values":image, "labels":label}


train_dataset.set_transform(train_hf_transform)
val_dataset.set_transform(val_hf_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCHSIZE, shuffle=False)

batch = next(iter(train_loader))
logger.info(batch["pixel_values"].shape)


###################################################################
# MODEL, OPTIM, CRITERION
###################################################################
model = ViT(
        in_height=IN_HEIGHT,
        in_width=IN_WIDTH,
        out_height=OUT_HEIGHT,
        out_width=OUT_WIDTH,
        in_channels=IN_CHANNELS,
        emb_dim=EM_DIM,
        num_heads=NUM_HEADS,
        mlp_size=MLP_SIZE,
        dropout=DROPOUT,
        att_dropout=ATT_DROPOUT,
        num_classes=NUM_CLASSES,
        num_layers=NUM_LAYERS
    ).to(DEVICE)
criterion = CrossEntropyLoss()
optim = Adam(model.parameters(), lr=LR)
scaler = GradScaler()   
writer = SummaryWriter()

logger.info(f"model, loss and optimizer are loaded")

###################################################################
# iou and miou
###################################################################
def iou(pred, target, classe):
    if pred.ndim > target.ndim:
        pred = pred.argmax(dim=1)

    pred_c = (pred == classe)
    target_c = (target == classe)

    intersection = (pred_c & target_c).sum().float()
    union = (pred_c | target_c).sum().float()
    
    return intersection / (union + 1e-8)

def miou(pred, target):
    
    return torch.mean(torch.stack([iou(pred, target, classes) for classes in range(NUM_CLASSES)]))

def accuracy1(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean()

def accuracy5(logits, targets, k):
    _, topk = logits.topk(k, dim=1)
    targets = targets.view(-1, 1)
    return (targets == topk).any(dim=1).float().mean()

#####################################################################
# VALIDATE
####################################################################
def validation_step(model, val_loader, device, criterion, k ):
    model.eval()
    
    val_loss = 0
    topk1 = 0
    topk5 = 0
    num_batches = 0
    with torch.no_grad():
        for batches in tqdm(val_loader, desc="Validation"):
            images = batches["pixel_values"].to(device)
            labels = batches["labels"].to(device)

            output = model(images)

            loss = criterion(output, labels)
            
            val_loss += loss.item()
            topk1 += accuracy1(output, labels).item()
            topk5 += accuracy5(output, labels, k).item()

            num_batches += 1

        return {
                "val_loss": val_loss / num_batches,
                "top1": topk1 / num_batches,
                "top5": topk5 / num_batches

                }



    
#####################################################################
# Training
#####################################################################

logger.info("Starting training...")

for epoch in range(EPOCHS):
    model.train()
    _loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        images, labels = batch["pixel_values"].to(DEVICE), batch["labels"].to(DEVICE)


        optim.zero_grad(set_to_none=True)

        with autocast():
            output = model(images)
            loss = criterion(output, labels)
        
        
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        _loss += loss.item()

    total_loss = _loss / len(train_loader)

    validation = validation_step(model, val_loader, DEVICE, criterion, k=5)
        
    logger.info(f"Epoch {epoch+1} |" 
                f"Training Loss: {total_loss:.4f}"
                f"Loss Val: {validation['val_loss']:.4f} |"
                f"Topk1 accuracy: {validation['top1']:.4f} |"
                f"Topk5 accuracy: {validation['top5']:.4f} |"
                    )

    writer.add_scalar("Loss/train", total_loss, epoch)
    writer.add_scalar("Loss/val", validation['val_loss'], epoch)
    writer.add_scalar("Accuracy/top1", validation['top1'], epoch)
    writer.add_scalar("Accuracy/top5", validation['top5'], epoch)

    if epoch % 10 == 0:
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "scaler": scaler.state_dict(),
            },
            f"vit_epoch_{epoch}.pt"
        )

writer.close()

        
        
        

        




























