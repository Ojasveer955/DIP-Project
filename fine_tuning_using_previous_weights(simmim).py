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

# Path to the unzipped images
DATA_DIR = './Dataset of Tuberculosis Chest X-rays Images/Dataset of Tuberculosis Chest X-rays Images/Dataset of Tuberculosis Chest X-rays Images'
CLASSES = ['Normal Chest X-rays', 'TB Chest X-rays']
LABEL_MAP = {k: v for v, k in enumerate(CLASSES)} # {'Normal': 0, 'TB': 1}

# --- Create the master DataFrame ---
image_paths = []
labels = []

for cls in CLASSES:
    class_path = os.path.join(DATA_DIR, cls)
    for img_name in os.listdir(class_path):
        image_paths.append(os.path.join(class_path, img_name))
        labels.append(LABEL_MAP[cls])

df = pd.DataFrame({'path': image_paths, 'label': labels})
print(f"Total images found: {len(df)}")

# --- Create the 80/20 Train/Validation Split ---
# We use stratify=df['label'] to ensure both classes are split proportionally
train_df, val_df = train_test_split(
    df,
    test_size=0.20,         # 20% for validation
    random_state=SEED,      # Use our reproducible seed
    stratify=df['label']    # Keep class balance
)

print(f"Training images: {len(train_df)}")
print(f"Validation images: {len(val_df)}")

# Check training set balance
print("\nTraining set distribution:")
print(train_df['label'].value_counts())

# Check validation set balance
print("\nValidation set distribution:")
print(val_df['label'].value_counts())

# --- 1. Define Transforms ---
# ImageNet mean/std (Swin model expects this)
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# Augmentations for the TRAINING set
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_std[0], std=mean_std[1]),
])

# Simple transforms for the VALIDATION set
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

BATCH_SIZE = 64 # You can tune this based on the HPC GPU

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

print(f"DataLoaders created with batch size {BATCH_SIZE}")

# -----------------------------------------------------------------
# <<< *** PLACEHOLDER: SET YOUR WEIGHTS PATH HERE *** >>>
# -----------------------------------------------------------------
# For SimMIM Notebook, use:
# WEIGHTS_PATH = "/content/drive/MyDrive/TB/simmim_swin_backbone.pth"

# For BYOL Notebook, use:
# WEIGHTS_PATH = "/content/drive/MyDrive/TB/byol_swin_backbone.pth"
#
# FOR YOUR FIRST RUN, CHOOSE ONE:
WEIGHTS_PATH = "./simmim_swin_backbone.pth" # <-- CHANGE THIS
# -----------------------------------------------------------------


# 1. Load the Swin Transformer "brain" (with NO mouth/head)
#    We set num_classes=0, which means "just give me the encoder".
model_backbone = timm.create_model(
    'swin_base_patch4_window7_224',
    pretrained=False,     # We are NOT using ImageNet this time
    num_classes=0         # <-- This gives us just the encoder
)

# 2. Load our NEW expert brain (the weights from the HPC)
#    This is the "brain transplant"
print(f"Loading pre-trained backbone weights from: {WEIGHTS_PATH}")

try:
    # Load the saved state dict from the .pth file
    original_state_dict = torch.load(WEIGHTS_PATH, map_location=device)

    # Get the keys from the model we want to load INTO
    model_keys = model_backbone.state_dict().keys()

    # Create a new, empty state dict to hold the "translated" keys
    new_state_dict = {}

    print("Translating SimMIM state_dict keys (e.g., 'layers_0' -> 'layers.0')...")

    # This is the translation logic:
    for k_orig, v in original_state_dict.items():
        k_new = k_orig
        if k_orig.startswith('layers_'):
            k_new = k_orig.replace('layers_', 'layers.')

        # Only add the key if it exists in our new model
        if k_new in model_keys:
            new_state_dict[k_new] = v

    # Load the "fixed" state dict. strict=False is important
    # as the .pth file might have extra keys (like a decoder).
    model_backbone.load_state_dict(new_state_dict, strict=False)
    print("Successfully loaded and translated self-supervised backbone weights!")

except Exception as e:
    print(f"Error loading weights: {e}")
    print("WARNING: MODEL IS USING RANDOM WEIGHTS. MAKE SURE YOUR .pth FILE EXISTS.")


# 3. "Add the classifier" (bolt on the new "mouth")
#    This is the step you asked about.
num_features = model_backbone.num_features # Get the brain's output size (1024)

# We create a new, final model that contains the
# brain + a new mouth
model = nn.Sequential(
    model_backbone,
    nn.Linear(num_features, 2) # <-- This nn.Linear layer IS the classifier
)

# Move the final model to the GPU
model = model.to(device)

print(f"\nFinal model created with a new classifier head (outputting 2 classes).")

# --- Define Training and Validation Functions ---

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Performs one full epoch of training."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Logging
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


@torch.no_grad() # Disable gradients for validation
def validate_one_epoch(model, dataloader, criterion, device):
    """Performs one full epoch of validation."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    for inputs, labels in tqdm(dataloader, desc="Validating"):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Get probabilities for AUROC
        probs = torch.softmax(outputs, dim=1)[:, 1] # Prob of class 1 (TB)
        preds = torch.argmax(outputs, dim=1)

        # Logging
        total_loss += loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    # Calculate AUROC
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError as e:
        print(f"AUROC calculation warning: {e}")
        auroc = 0.5 # Default value if only one class is present

    return avg_loss, accuracy, auroc

print("Training and validation functions defined.")

# --- Setup for Training ---
NUM_EPOCHS = 20  # We fine-tune for fewer epochs than pre-training
LEARNING_RATE = 1e-4

# We only want to train the "mouth" (classifier).
# But, it's also common to "fine-tune" the brain (backbone) with a
# very small learning rate.

# For this experiment, let's fine-tune the whole model.
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# We use CrossEntropyLoss for classification
criterion = nn.CrossEntropyLoss()

# Path to save the best model
BEST_MODEL_SAVE_PATH = "./fine_tuned_best_SIMMiM.pth"
best_auroc = 0.0

print("--- Starting Fine-Tuning ---")

# --- Main Training Loop ---
for epoch in range(NUM_EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")

    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Epoch {epoch+1} Training:   Loss={train_loss:.4f}, Accuracy={train_acc:.4f}")

    val_loss, val_acc, val_auroc = validate_one_epoch(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1} Validation: Loss={val_loss:.4f}, Accuracy={val_acc:.4f}, AUROC={val_auroc:.4f}")

    # Save the best model based on validation AUROC
    if val_auroc > best_auroc:
        print(f"New best model! AUROC improved from {best_auroc:.4f} to {val_auroc:.4f}.")
        best_auroc = val_auroc
        torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
        print(f"Best model saved to {BEST_MODEL_SAVE_PATH}")

print("--- Fine-Tuning Finished ---")
print(f"Best Validation AUROC achieved: {best_auroc:.4f}")

# import matplotlib.pyplot as plt
# import seaborn as sns

print("\n--- Loading best model for final evaluation ---")
# 1. Re-create the model architecture
#    (We must do this in case we are in a new session)
model_backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=0)
num_features = model_backbone.num_features
final_model = nn.Sequential(model_backbone, nn.Linear(num_features, 2))

# 2. Load the best fine-tuned weights
final_model.load_state_dict(torch.load(BEST_MODEL_SAVE_PATH))
final_model = final_model.to(device)
final_model.eval() # Set to evaluation mode

print("Best model loaded successfully.")

# --- 3. Run final evaluation on the validation set ---
all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for inputs, labels in tqdm(val_loader, desc="Final Evaluation"):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = final_model(inputs)
        probs = torch.softmax(outputs, dim=1)[:, 1] # Prob of class 1 (TB)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# --- 4. Calculate and Print Final Metrics ---
final_accuracy = accuracy_score(all_labels, all_preds)
final_auroc = roc_auc_score(all_labels, all_probs)
cm = confusion_matrix(all_labels, all_preds)

print(f"\n--- Final Validation Metrics ---")
print(f"Accuracy: {final_accuracy:.4f}")
print(f"AUROC:    {final_auroc:.4f}")
print("----------------------------------")

"""
# --- 5. Plot Confusion Matrix ---
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title(f'Final Confusion Matrix (AUROC: {final_auroc:.4f})')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
"""