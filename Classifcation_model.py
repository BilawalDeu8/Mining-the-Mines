# Importing necessary libraries
import os
import random as rnd
import tifffile as tiff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import timm

data_dir = "/path"
train_data_dir = data_dir + '/path'
test_data_dir = data_dir + '/path'

train_data_df = pd.read_csv(f'{data_dir}train/answer.csv', names=["file_path", "label"], header=None)

# Data augmentation transforms
train_transforms = v2.Compose([
    v2.ToTensor(),
    v2.RandomRotation(degrees=90),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.ToDtype(torch.float32, scale=False),
])

val_transforms = v2.Compose([
    v2.ToTensor(),
    v2.ToDtype(torch.float32, scale=False),
])

# Feature engineering function
def preprocess_image(image_data):
    band7 = image_data[..., 7]
    band8 = image_data[..., 8]
    band10 = image_data[..., 10]
    band11 = image_data[..., 11]
    endmi = ((band7 + band8) - (band10 + band11)) / (band7 + band8 + band10 + band11 + 1e-10)
    band3 = image_data[..., 3]
    band7 = image_data[..., 7]
    ndvi = (band7 - band3) / (band7 + band3 + 1e-10)

    processed_image = np.concatenate([
        image_data * 2 - 1,  # scale band data from -1 to 1
        np.expand_dims(endmi, axis=-1),
        np.expand_dims(ndvi, axis=-1),
    ], axis=-1)
    return processed_image


class CustomTrainDataset(Dataset):
    def __init__(self,folder,data,transforms):
        self.folder = folder
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.loc[index]
        image_path = f"{self.folder}{data_row.file_path}"
        image_data = tiff.imread(image_path)
        image_data = preprocess_image(np.array(image_data))
        image_tensor = self.transforms(image_data)
        label = data_row.label

        return {"image": image_tensor, "label": torch.tensor(label, dtype=torch.long)}

# Evaluation metrics
from sklearn.metrics import f1_score, confusion_matrix

def find_optimal_threshold(targets, predictions):
    base_f1 = f1_score(targets, predictions > 0.5)
    best_f1 = 0
    best_th = -1
    for threshold in [i / 100 for i in range(100)]:
        curr_f1 = f1_score(targets, predictions > threshold)
        if curr_f1 > best_f1:
            best_f1 = curr_f1
            best_th = threshold

    tn, fp, fn, tp = confusion_matrix(targets.numpy(), predictions.numpy() > best_th).ravel()
    print(f"tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}")
    return base_f1, best_f1, best_th

class style:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    END = '\033[0m'
    BOLD = '\033[1m'

def set_seed(seed=42):
    rnd.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# Evaluation function
def evaluate_model(cfg, model, data_loader, epoch=-1):
    loss_fn = nn.CrossEntropyLoss(weight=cfg.weights.to(cfg.device), label_smoothing=0.1)

    model.eval()
    val_loss = 0

    targets = []
    predictions = []

    data_loader_length = len(data_loader)
    progress_bar = tqdm(enumerate(data_loader), total=data_loader_length)
    for step, data in progress_bar:
        input_data = data["image"].to(cfg.device, non_blocking=True)
        target_labels = data["label"].to(cfg.device, non_blocking=True)

        with torch.no_grad():
            logits = model(input_data)

        loss = loss_fn(logits, target_labels)
        val_loss += loss.item()

        targets.append(target_labels.detach().cpu())
        predictions.append(logits.detach().cpu())

    targets = torch.cat(targets, dim=0)
    predictions = torch.cat(predictions, dim=0)
    predictions = F.sigmoid(predictions)

    val_loss /= data_loader_length
    base_f1, best_f1, best_th = find_optimal_threshold(targets, predictions[:, 1])

    print(f'Epoch {epoch} validation loss = {val_loss:.4f}, base f1 score (0.5 threshold) = {base_f1:.4f} (best threshold: {best_th:.2f} -> f1 {best_f1:.4f})')

    predictions = (predictions[:, 1] > best_th).int()
    accuracy = (predictions == targets).float().mean()
    print(f'Accuracy: {accuracy:.4f}')

    confusion_matrix_values = torch.zeros(2, 2, dtype=torch.int32)
    for pred, tgt in zip(predictions, targets):
        confusion_matrix_values[pred, tgt] += 1
    print('Confusion Matrix:')
    print(confusion_matrix_values)
    return val_loss, best_f1

# Training function
def train_epoch(cfg, model, train_loader, optimizer, scheduler, epoch):
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.apex)
    loss_fn = nn.CrossEntropyLoss(weight=cfg.weights.to(cfg.device), label_smoothing=0.1)

    model.train()
    train_loss = 0
    lr_history = []

    targets = []
    predictions = []

    data_loader_length = len(train_loader)
    progress_bar = tqdm(enumerate(train_loader), total=data_loader_length)
    for step, data in progress_bar:
        input_data = data["image"].to(cfg.device, non_blocking=True)
        target_labels = data["label"].to(cfg.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=cfg.apex):
            logits = model(input_data)
            loss = loss_fn(logits, target_labels)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.clip_val)

        train_loss += loss.item()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if scheduler is None:
            lr = optimizer.param_groups[0]['lr']
        else:
            scheduler.step()
            lr = scheduler.get_last_lr()[0]

        progress_bar.set_description(f"Epoch {epoch} training {step+1}/{data_loader_length} [LR {lr:.6f}] - loss: {train_loss/(step+1):.4f}")
        lr_history.append(lr)

        targets.append(target_labels.detach().cpu())
        predictions.append(logits.detach().cpu())

        targets = torch.cat(targets, dim=0)
        predictions = torch.cat(predictions, dim=0)
        predictions = F.sigmoid(predictions)

        train_loss /= data_loader_length
        base_f1, best_f1, best_th = find_optimal_threshold(targets, predictions[:, 1])

        print(f'Epoch {epoch} train loss = {train_loss:.4f}, base f1 score (0.5 threshold) = {base_f1:.4f} (best threshold: {best_th:.2f} -> f1 {best_f1:.4f})')
        return train_loss, best_f1, lr_history

# Configuration class
class CFG:
    seed = 42
    num_folds = 5
    train_folds = [0, 1]  # [0, 1, 2, 3, 4]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    apex = True  # use half precision

    model_name = "maxvit_tiny_tf_512"
    epochs = 10
    weights = torch.tensor([0.206119, 0.793881], dtype=torch.float32)

    clip_val = 1000.
    batch_size = 20
    gradient_accumulation_steps = 1

    lr = 1e-4
    weight_decay = 1e-2

# K-Fold cross-validation
from sklearn.model_selection import StratifiedKFold

set_seed(CFG.seed)
skf = StratifiedKFold(n_splits=CFG.num_folds, random_state=CFG.seed, shuffle=True)
for fold, (train_idx, test_idx) in enumerate(skf.split(train_data_df["file_path"].values, train_data_df["label"].values)):
    train_data_df.loc[test_idx, "fold"] = fold

# Train and evaluate model for each fold
for FOLD in CFG.train_folds:
    set_seed(CFG.seed)

    # Prepare data
    fold_train_data = train_data_df[train_data_df["fold"] != FOLD].reset_index(drop=True)
    fold_valid_data = train_data_df[train_data_df["fold"] == FOLD].reset_index(drop=True)

    print("Data distribution:")
    display(pd.merge(
        fold_valid_data.groupby(by=["label"])["file_path"].count().rename("valid").reset_index(),
        fold_train_data.groupby(by=["label"])["file_path"].count().rename("train").reset_index(),
        on="label", how="left"
    ).T)

    train_dataset = CustomTrainDataset(train_data_dir, fold_train_data, transforms=train_transforms)
    valid_dataset = CustomTrainDataset(train_data_dir, fold_valid_data, transforms=val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # Prepare model, optimizer, and scheduler
    model = timm.create_model(CFG.model_name, in_chans=14, num_classes=2, pretrained=True)
    model = model.to(CFG.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):_}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, eta_min=1e-6, T_max=CFG.epochs * len(train_loader),
    )

    # Train and evaluate
    lr_history = []
    train_loss_history = []
    train_score_history = []
    val_loss_history = []
    val_score_history = []

    best_score = 0
    for epoch in range(CFG.epochs):
        train_loss, train_score, train_lr = train_epoch(CFG, model, train_loader, optimizer, scheduler, epoch)
        train_loss_history.append(train_loss)
        train_score_history.append(train_score)
        lr_history.extend(train_lr)

        val_loss, val_score = evaluate_model(CFG, model, valid_loader, epoch)
        val_loss_history.append(val_loss)
        val_score_history.append(val_score)

        if val_score > best_score:
            print(f"{TerminalStyle.GREEN}New best score: {best_score:.4f} -> {val_score:.4f}{TerminalStyle.END}")
            best_score = val_score
            torch.save(model.state_dict(), f'{data_dir}best_model_fold/best_model_fold_{FOLD}.pth')

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].plot(train_loss_history, label="Train")
    axes[0].plot(val_loss_history, label="Valid")
    axes[0].title.set_text("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(train_score_history, label="Train")
    axes[1].plot(val_score_history, label="Valid")
    axes[1].title.set_text("F1 score")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(lr_history, label="LR")
    axes[2].legend()
    axes[2].title.set_text("Learning rate")
    axes[2].set_xlabel("Step")
    fig.suptitle(f"Fold {FOLD}")
    fig.tight_layout()
    plt.show()

# Inference on test data
class ImageTestDataset(Dataset):
    def __init__(self, folder, file_names, transforms=None):
        self.folder = folder
        self.file_names = file_names
        self.transforms = transforms

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        image_path = f"{self.folder}{self.file_names[index]}"
        image_data = tiff.imread(image_path)
        image_data = preprocess_image(image_data)
        image_tensor = self.transforms(image_data)
        return {"image": image_tensor}

def inference(cfg, model, data_loader):
    model.eval()
    predictions = []

    data_loader_length = len(data_loader)
    progress_bar = tqdm(enumerate(data_loader), total=data_loader_length)
    for step, data in progress_bar:
        input_data = data["image"].to(cfg.device, non_blocking=True)
        with torch.no_grad():
            logits = model(input_data)

        predictions.append(logits.detach().cpu())
        
    predictions = torch.cat(predictions, dim=0)
    predictions = F.sigmoid(predictions)
    return predictions[:, 1]

test_data_df = pd.read_csv(f'{data_dir}uploadsample.csv', names=["file_name", "label"], header=None)
test_data_df["probability"] = 0

test_dataset = ImageTestDataset(test_data_dir, test_data_df["file_name"].values, test_transforms)
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=False,
    drop_last=False,
)

set_seed(CFG.seed)
all_predictions = []

for FOLD in CFG.train_folds:
    print(f"Inference for fold: {FOLD}")
    model = timm.create_model(CFG.model_name, in_chans=14, num_classes=2, pretrained=False)
    model = model.to(CFG.device)

    model.load_state_dict(torch.load(f"{data_dir}best_model_fold/best_model_fold_{FOLD}.pth", map_location=CFG.device))
    predictions = inference(CFG, model, test_loader)

    all_predictions.append(predictions.numpy())

    # Average predictions from fold models
    test_data_df["probability"] += predictions.numpy() / len(CFG.train_folds)

all_predictions = np.concatenate(all_predictions)

threshold = 0.5  # Adjust the threshold as needed
predicted_labels = (all_predictions > threshold).astype(int)
accuracy = np.mean(predicted_labels == test_data_df["label"].values)
print(f"Accuracy on test set: {accuracy:.4f}") #Not correct

# Save predictions
THRESHOLD = 0.8
test_data_df["label"] = (test_data_df["probability"] > THRESHOLD).astype(int)
print("Positive predictions:", test_data_df["label"].sum())
test_data_df[["file_name", "label"]].to_csv("predictions.csv", index=False, header=False)

