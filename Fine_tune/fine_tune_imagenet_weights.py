import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
import pandas as pd
import numpy as np
import os
import time
import random
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from tqdm.auto import tqdm
import copy
import sys

# --- 1. Script Configuration ---
EXPERIMENT_NAME = 'ImageNet Baseline'
BEST_MODEL_SAVE_PATH = "./Imagenet_weights_fine_tuned_model.pth"
# ---------------------------------

# GPU setup and reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    device = torch.device("cpu")
    print("Using CPU.")

print(f"--- RUNNING EXPERIMENT: {EXPERIMENT_NAME} ---")
print(f"Best model will be saved to: {BEST_MODEL_SAVE_PATH}")

# Path to the unzipped images (using the 3-level path from your working logs)
DATA_DIR = './Dataset of Tuberculosis Chest X-rays Images/Dataset of Tuberculosis Chest X-rays Images/Dataset of Tuberculosis Chest X-rays Images'
CLASSES = ['Normal Chest X-rays', 'TB Chest X-rays']
LABEL_MAP = {k: v for v, k in enumerate(CLASSES)} # {'Normal': 0, 'TB': 1}

# --- Create the master DataFrame ---
image_paths = []
labels = []

print(f"Loading file paths from {DATA_DIR}...")
for cls in CLASSES:
    class_path = os.path.join(DATA_DIR, cls)
    if not os.path.isdir(class_path):
         print(f"Error: Directory not found at {class_path}")
         print(f"Please double-check the DATA_DIR path in this script.")
         sys.exit() # Exit the script if data isn't found

    for img_name in os.listdir(class_path):
        image_paths.append(os.path.join(class_path, img_name))
        labels.append(LABEL_MAP[cls])

df = pd.DataFrame({'path': image_paths, 'label': labels})
print(f"Total images found: {len(df)}")


# --- Create the 80/10/10 Train/Validation/Test Split ---
print("\n--- Creating 80/10/10 Train/Val/Test Split ---")
# First, split into 80% train and 20% temporary (for val/test)
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,         # 20% for val/test
    random_state=SEED,      # Use our reproducible seed
    stratify=df['label']    # Keep class balance
)

# Now, split the 20% temporary set in half (50% of 20% = 10% of total)
# to get 10% validation and 10% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,         # Split the 20% into two 10% chunks
    random_state=SEED,      # Use the same seed
    stratify=temp_df['label'] # Keep class balance
)

print(f"Total images: {len(df)}")
print(f"Training images:   {len(train_df)} (80%)")
print(f"Validation images: {len(val_df)} (10%)")
print(f"Test images:       {len(test_df)} (10%)")

# --- 1. Define Transforms ---
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_std[0], std=mean_std[1]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_std[0], std=mean_std[1]),
])

# --- 2. Define the Dataset Class ---
class TuberculosisDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['path']
        label = self.df.iloc[idx]['label']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# --- 3. Create Datasets and DataLoaders ---
train_dataset = TuberculosisDataset(train_df, transform=train_transform)
val_dataset = TuberculosisDataset(val_df, transform=val_transform)
test_dataset = TuberculosisDataset(test_df, transform=val_transform) # <-- TEST DATASET

BATCH_SIZE = 64 # Using 32 to be safe and consistent with other runs

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2) # <-- TEST LOADER

print(f"DataLoaders created with batch size {BATCH_SIZE}")


# -----------------------------------------------------------------
# <<< *** 3. THIS IS THE ONLY MODEL-LOADING CHANGE *** >>>
# -----------------------------------------------------------------
print("Loading ImageNet-pretrained backbone weights...")
model_backbone = timm.create_model(
    'swin_base_patch4_window7_224',
    pretrained=True,     # <-- This is the key: load ImageNet weights
    num_classes=0        # <-- This gives us just the encoder
)
print("Successfully loaded ImageNet-pretrained backbone!")
# -----------------------------------------------------------------


# 4. "Add the classifier" (bolt on the new "mouth")
num_features = model_backbone.num_features
model = nn.Sequential(
    model_backbone,
    nn.Linear(num_features, 2)
)
model = model.to(device)
print(f"\nFinal model created with a new classifier head (outputting 2 classes).")

# --- Define Training and Validation Functions ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    # Note: leave=False for tqdm in a script
    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device, desc="Validating"):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    # Note: leave=False for tqdm in a script
    for inputs, labels in tqdm(dataloader, desc=desc, leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)
        total_loss += loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError as e:
        print(f"AUROC calculation warning: {e}")
        auroc = 0.5
    return avg_loss, accuracy, auroc

print("Training and validation functions defined.")

# --- Setup for Training ---
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
best_auroc = 0.0

print("--- Starting Fine-Tuning ---")

# --- Main Training Loop ---
for epoch in range(NUM_EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
    
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Epoch {epoch+1} Training:   Loss={train_loss:.4f}, Accuracy={train_acc:.4f}")
    
    # We use the VALIDATION loader to pick the best model
    val_loss, val_acc, val_auroc = validate_one_epoch(model, val_loader, criterion, device, desc="Validating")
    print(f"Epoch {epoch+1} Validation: Loss={val_loss:.4f}, Accuracy={val_acc:.4f}, AUROC={val_auroc:.4f}")

    if val_auroc > best_auroc:
        print(f"New best model! AUROC improved from {best_auroc:.4f} to {val_auroc:.4f}.")
        best_auroc = val_auroc
        torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
        print(f"Best model saved to {BEST_MODEL_SAVE_PATH}")

print("--- Fine-Tuning Finished ---")
print(f"Best Validation AUROC achieved during training: {best_auroc:.4f}")

# -----------------------------------------------------------------
# <<< *** 4. FINAL, UNBIASED TEST *** >>>
# -----------------------------------------------------------------
print("\n--- Loading best model for FINAL TEST evaluation ---")
# 1. Re-create the model architecture
model_backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=0)
num_features = model_backbone.num_features
final_model = nn.Sequential(model_backbone, nn.Linear(num_features, 2))

# 2. Load the best fine-tuned weights
final_model.load_state_dict(torch.load(BEST_MODEL_SAVE_PATH))
final_model = final_model.to(device)
final_model.eval()
print("Best model loaded successfully.")

# 3. Run final evaluation on the TEST SET
print("Running final evaluation on the 10% (unseen) TEST set...")
# We call our function, but this time passing the test_loader
test_loss, test_acc, test_auroc = validate_one_epoch(
    final_model, 
    test_loader,  # <-- Using the locked-box test_loader
    criterion, 
    device, 
    desc="Final Test"
)

# 4. Calculate and Print Final Metrics
print(f"\n--- FINAL TEST SET METRICS (The 'Paper' Result) ---")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test AUROC:    {test_auroc:.4f}")
print("------------------------------------------------------")
