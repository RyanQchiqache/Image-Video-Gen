import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision



from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torchvision.transforms as T 

from vision_transformers import ViT
from datasets import load_dataset
from loguru import logger
from torch.cuda.amp import GradScaler, autocast 
from tqdm import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################
# CONFIG
###############################################################################
EPOCHS=100
BATCHSIZE=100
LR=5e-4
RESIZE= (320, 320)

#################################################################################
# ViT CONFIG
################################################################################
IN_HEIGHT=320
IN_WIDTH=320
OUT_HEIGHT= IN_HEIGHT // 16
OUT_WIDTH= IN_WIDTH // 16
DROPOUT=0.1
ATT_DROPOUT=0.1
NUM_CLASSES=1000
IN_CHANNELS=3
NUM_HEADS=8
MLP_SIZE=1536 
EM_DIM=384
NUM_LAYERS=6


################################################################################
# DATASET AND TRANSFORMATION
################################################################################
train_transform = T.Compose([
        T.RandomHorizontalFlip(0.5),
        T.Resize(RESIZE),
        T.ToTensor(),
        T.Normalize(
            mean=[0.5 for _ in range(IN_CHANNELS)],
            std=[0.5 for _ in range(IN_CHANNELS)]
    ),
])

val_transform = T.Compose([
        T.Resize(RESIZE),
        T.ToTensor(),
        T.Normalize(
            mean=[0.5 for _ in range(IN_CHANNELS)],
            std=[0.5 for _ in range(IN_CHANNELS)]
    ),
])

def train_hf_transform(example):
    image = example["image"]
    label = example["label"]
    image = train_transform(image)
    return {"pixel_values":image, "labels":label}

def val_hf_transform(example):
    image = example["image"]
    label = example["label"]
    image = val_transform(image)
    return {"pixel_values":image, "labels":label}


train_dataset = load_dataset(
    "frgfm/imagenette",
    "full_size",
    split="train"
)

val_dataset = load_dataset(
    "frgfm/imagenette",
    "full_size",
    split="validation"
)

logger.info(f"train/val datasets have been loaded!!")

train_dataset.set_transform(train_hf_transform)
val_dataset.set_transform(val_hf_transform)

train_dataset.set_format(type="torch")
val_dataset.set_format(type="torch")

logger.info(f"set_transform OK!")

train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCHSIZE, shuffle=False)
batch = next(iter(train_loader))
logger.info(f"size batch: {batch["pixel_values"].shape}")
logger.info(f"train/val DataLoader OK!!")


###################################################################################
# LOSS, CRITERION, OPTIM 
###################################################################################

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

logger.info(f"model, loss, criterion, criterion, optim  OK!!")


######################################################################################
# METRICS
#####################################################################################

def accuracy1(logits, labels):
    logits = logits.argmax(dim=1)
    return (logits == labels).float().mean()

def accuracy5(logits, labels, k):
    _, top_k = logits.topk(k, dim= 1)
    labels = labels.view(-1,1)
    return (top_k == labels).any(dim=1).float().mean()


#####################################################################################
# VALIDATION 
####################################################################################

def validation_step(model, val_loader, criterion, device, k):
    model.eval()

    val_loss = 0 
    topk1 = 0 
    topk5 = 0 
    num_batches = 0 

    with torch.no_grad():
        for batches in tqdm(val_loader, desc="Validation"):
            image = batches["pixel_values"].to(device)
            label = batches["labels"].to(device)

            output = model(image)

            loss = criterion(output, label)

            val_loss += loss.item()

            topk1 += accuracy1(output, label).item()
            topk5 += accuracy5(output, label, k).item()

            num_batches += 1

        return {
            "val_loss":val_loss / num_batches,
            "topk1": topk1 / num_batches,
            "topk5": topk5 / num_batches,
        }

            

#####################################################################
# TRAINING
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

