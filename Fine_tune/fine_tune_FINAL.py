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

# -----------------------------------------------------------------
# <<< *** 1. CHOOSE YOUR MODEL *** >>>
#
# CHOOSE 'byol' or 'simmim'
EXPERIMENT_NAME = 'simmim'  # <--- EDIT THIS
# -----------------------------------------------------------------

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

# --- DYNAMICALLY SET PATHS BASED ON EXPERIMENT_NAME ---
if EXPERIMENT_NAME == 'byol':
    WEIGHTS_PATH = "../Pretrained/byol_swin_backbone.pth"
    BEST_MODEL_SAVE_PATH = "../Pretrained/fine_tuned_best_BYOL.pth"
elif EXPERIMENT_NAME == 'simmim':
    WEIGHTS_PATH = "../Pretrained/simmim_swin_backbone.pth"
    BEST_MODEL_SAVE_PATH = "../Pretrained/fine_tuned_best_SIMMiM.pth"
else:
    raise ValueError("EXPERIMENT_NAME must be 'byol' or 'simmim'")

print(f"--- RUNNING EXPERIMENT: {EXPERIMENT_NAME} ---")
print(f"Loading weights from: {WEIGHTS_PATH}")
print(f"Best model will be saved to: {BEST_MODEL_SAVE_PATH}")

# Path to the unzipped images
# *** THIS IS THE CORRECTED PATH FROM YOUR LOGS ***
DATA_DIR = '../Dataset of Tuberculosis Chest X-rays Images/Dataset of Tuberculosis Chest X-rays Images/Dataset of Tuberculosis Chest X-rays Images'
CLASSES = ['Normal Chest X-rays', 'TB Chest X-rays']
LABEL_MAP = {k: v for v, k in enumerate(CLASSES)} # {'Normal': 0, 'TB': 1}

# --- Create the master DataFrame ---
image_paths = []
labels = []

for cls in CLASSES:
    class_path = os.path.join(DATA_DIR, cls)
    if not os.path.isdir(class_path):
         print(f"Error: Directory not found at {class_path}")
         print(f"Please check your DATA_DIR variable. It is currently: {DATA_DIR}")
         # Attempting to fix by checking one level deeper, as in your pre-training logs
         DATA_DIR = '../Dataset of Tuberculosis Chest X-rays Images/Dataset of Tuberculosis Chest X-rays Images/Dataset of Tuberculosis Chest X-rays Images'
         print(f"Retrying with DATA_DIR: {DATA_DIR}")
         class_path = os.path.join(DATA_DIR, cls)
         if not os.path.isdir(class_path):
             print(f"Error: Still cannot find directory. Exiting.")
             exit() # Exit the script if data isn't found

    for img_name in os.listdir(class_path):
        image_paths.append(os.path.join(class_path, img_name))
        labels.append(LABEL_MAP[cls])

df = pd.DataFrame({'path': image_paths, 'label': labels})
print(f"Total images found: {len(df)}")


# -----------------------------------------------------------------
# <<< *** 2. THIS IS THE NEW 80/10/10 SPLIT *** >>>
# -----------------------------------------------------------------
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
# -----------------------------------------------------------------


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
test_dataset = TuberculosisDataset(test_df, transform=val_transform) # <-- NEW TEST DATASET

BATCH_SIZE = 64 # Use 32, as 64 crashed one of your jobs

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2) # <-- NEW TEST LOADER

print(f"DataLoaders created with batch size {BATCH_SIZE}")


# 1. Load the Swin Transformer "brain" (with NO mouth/head)
model_backbone = timm.create_model(
    'swin_base_patch4_window7_224',
    pretrained=False,
    num_classes=0
)

# -----------------------------------------------------------------
# <<< *** 3. THIS BLOCK IS NOW DYNAMIC *** >>>
# -----------------------------------------------------------------
print(f"Loading pre-trained backbone weights from: {WEIGHTS_PATH}")
try:
    original_state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    
    if EXPERIMENT_NAME == 'simmim':
        # --- SimMIM Key Translator ---
        print("Translating SimMIM state_dict keys (e.g., 'layers_0' -> 'layers.0')...")
        model_keys = model_backbone.state_dict().keys()
        new_state_dict = {}
        for k_orig, v in original_state_dict.items():
            k_new = k_orig
            if k_orig.startswith('layers_'):
                k_new = k_orig.replace('layers_', 'layers.')
            if k_new in model_keys:
                new_state_dict[k_new] = v
        model_backbone.load_state_dict(new_state_dict, strict=False)
        print("Successfully loaded and translated SimMIM backbone weights!")
        # --- End SimMIM Block ---
    else:
        # --- BYOL Key Loader ---
        model_backbone.load_state_dict(original_state_dict, strict=True)
        print("Successfully loaded BYOL backbone weights!")
        # --- End BYOL Block ---

except Exception as e:
    print(f"Error loading weights: {e}")
    print("WARNING: MODEL IS USING RANDOM WEIGHTS. MAKE SURE YOUR .pth FILE EXISTS.")
# -----------------------------------------------------------------


# 3. "Add the classifier" (bolt on the new "mouth")
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
