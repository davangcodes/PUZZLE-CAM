# ✅ Essential Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import json
import base64
from io import BytesIO
from tqdm import tqdm
import wandb
import os
from sklearn.metrics import average_precision_score
import torch.backends.cudnn as cudnn
from torchvision import transforms, models
import cv2

# ✅ Setup W&B and CUDA
wandb.login(key="a8832575b3abf5340a76141f84f38cb6c1c19247")  # Track experiments
cudnn.benchmark = True  # Speed up training if input sizes are consistent
wandb.init(project="resnet-puzzlecam", name="resnet50-puzzlecam-multilabel")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Puzzle-CAM ResNet50 Model Definition
class ResNetPuzzleCAM(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Load pretrained ResNet50 and remove final FC and avgpool layers
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Up to conv5_x
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)
        pooled = self.avgpool(feats)
        return self.classifier(pooled)

# ✅ Puzzle-CAM KL Divergence Loss: aligns patch predictions with full image
def puzzle_loss(full_logits, patch_logits):
    full_probs = torch.sigmoid(full_logits.detach())  # Prevent gradients flowing back
    loss = 0
    for patch_logit in patch_logits:
        patch_probs = torch.sigmoid(patch_logit)
        # KL divergence between patch and full image predictions
        loss += nn.functional.kl_div(patch_probs.log(), full_probs, reduction='batchmean')
    return loss / len(patch_logits)

# ✅ Splits a batch of images into 4 non-overlapping 2x2 patches
def split_into_patches(batch):
    B, C, H, W = batch.shape
    patches = []
    h_mid, w_mid = H // 2, W // 2
    for i in range(2):
        for j in range(2):
            patch = batch[:, :, i * h_mid:(i + 1) * h_mid, j * w_mid:(j + 1) * w_mid]
            patches.append(patch)
    return patches  # List of 4 patches

# ✅ Data Augmentation (only for training)
class AugmentationPipeline:
    def __init__(self, base_transform):
        self.base_transform = base_transform
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(0.4),
            transforms.ColorJitter(brightness=0.1, contrast=0.2),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            base_transform
        ])

    def __call__(self, img):
        return self.augmentations(img)

# ✅ CLAHE to enhance contrast of surgical images
def apply_clahe(image_np):
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

# ✅ Custom Dataset for CholecT45 in Base64 JSON format
class CholecT45Dataset(Dataset):
    def __init__(self, json_file, preprocess, augment=False):
        with open(json_file, 'r') as f:
            data = json.load(f)
        # Only keep samples that contain both image and verb labels
        self.data = [item for item in data if 'image' in item and 'verb_labels' in item]
        self.preprocess = preprocess
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Decode base64 image and apply CLAHE
        image = Image.open(BytesIO(base64.b64decode(item['image']))).convert("RGB")
        image_np = np.array(image)
        clahe_image = apply_clahe(image_np)
        image_pil = Image.fromarray(clahe_image)
        # Apply augmentation if training
        image_final = self.preprocess(image_pil) if not self.augment else AugmentationPipeline(self.preprocess)(image_pil)
        labels = torch.tensor(item['verb_labels'], dtype=torch.float32)
        return image_final, labels

# ✅ Base image transform (resize + tensor conversion)
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ✅ Load Train and Validation Datasets
train_dataset = CholecT45Dataset("../../../../instrument_verb_train.json", base_transform, augment=True)
val_dataset = CholecT45Dataset("../../../../instrument_verb_val.json", base_transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=12, pin_memory=True)

# ✅ Initialize Model
model = ResNetPuzzleCAM().to(device)

# ✅ Weighted BCE Loss to handle class imbalance
loss_weights = torch.tensor([0.38, 0.06, 0.07, 0.84, 1.21, 2.52, 1.30, 6.88, 17.07, 0.45], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weights)

# ✅ Optimizer and Scheduler
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-4, total_steps=len(train_loader) * 30,
    pct_start=0.1, anneal_strategy="cos", div_factor=10, final_div_factor=100
)

# ✅ Directory for saving weights
os.makedirs("weights", exist_ok=True)

# ✅ Evaluation Function: Computes Loss + mAP
def evaluate_model(model, loader, criterion, device):
    model.eval()
    all_targets, all_preds = [], []
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            all_targets.append(labels.cpu().numpy())
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
    return running_loss / len(loader), average_precision_score(
        np.concatenate(all_targets), np.concatenate(all_preds), average="macro"
    )

# ✅ Training Loop
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30, save_dir="weights/"):
    best_val_mAP = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            full_logits = model(images)  # Full image logits
            patches = split_into_patches(images)  # 4 patch views
            patch_logits = [model(p) for p in patches]  # Forward each patch
            bce = criterion(full_logits, labels)  # Main loss
            pzl = puzzle_loss(full_logits, patch_logits)  # Puzzle-CAM KL loss
            loss = bce + 0.4 * pzl  # Combine losses with weighted Puzzle Loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        val_loss, val_mAP = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}: Train Loss={running_loss/len(train_loader):.4f}, Val Loss={val_loss:.4f}, Val mAP={val_mAP:.4f}")
        # Save best model
        if val_mAP > best_val_mAP:
            best_val_mAP = val_mAP
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
        # W&B logging
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": running_loss / len(train_loader),
            "val_loss": val_loss,
            "val_mAP": val_mAP
        })

# ✅ Resume from checkpoint if exists
checkpoint_path = os.path.join("weights", "best_model.pth")
if os.path.exists(checkpoint_path):
    print(f"\n🔁 Resuming from checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path))

# ✅ Start Training
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30)
