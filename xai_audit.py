import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
import pandas as pd
import numpy as np
import os
import time
import random
import sys
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2
import warnings

# --- 1. MATPLOTLIB SETUP FOR HPC (SAVE TO FILE) ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- 2. Grad-CAM Imports ---
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    print("Error: 'grad-cam' library not found.")
    print("Please install it: pip install grad-cam")
    sys.exit()

# --- 3. Configuration ---
warnings.filterwarnings('ignore')
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU.")

# --- Paths ---
# Use the correct data path from your working logs
DATA_DIR = './Dataset of Tuberculosis Chest X-rays Images/Dataset of Tuberculosis Chest X-rays Images/Dataset of Tuberculosis Chest X-rays Images'
OUTPUT_DIR = "./xai_final_results"
NUM_IMAGES_TO_AUDIT = 10 # Let's generate 10 examples

# --- THIS IS THE LIST OF MODELS WE WILL TEST ---
# We use the paths from your 'ls' command
models_to_audit = {
    "ImageNet": "./Fine_tune/Imagenet_weights_fine_tuned_model.pth",
    "BYOL": "./Pretraining/byol_swin_backbone.pth",
    "SimMIM": "./Pretraining/simmim_swin_backbone.pth"
}
# -----------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"XAI audit images will be saved to: {OUTPUT_DIR}")

# --- 4. Helper Classes and Functions ---

def tensor_to_rgb_image(tensor):
    """Converts a normalized tensor to a 0-1 range numpy image for Grad-CAM."""
    tensor = tensor.clone().detach().cpu()
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    tensor = tensor * std[:, None, None] + mean[:, None, None]
    tensor = tensor.clamp(0, 1)
    rgb_img = tensor.permute(1, 2, 0).numpy() # (H, W, C)
    return rgb_img

class TB_Base_Dataset(Dataset):
    """Returns the transformed tensor and the label."""
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
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)
            
        # Return tensor, label, and the original path for saving
        return image_tensor, torch.tensor(label, dtype=torch.long), img_path

# --- 5. Load Data (80/10/10 split) ---
print("Loading dataset...")
CLASSES = ['Normal', 'Tuberculosis']
LABEL_MAP = {k: v for v, k in enumerate(CLASSES)}

image_paths = []
labels = []

# This block checks for the 3-level and 1-level path
try:
    for cls in ['Normal Chest X-rays', 'TB Chest X-rays']:
        class_path = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(class_path):
             DATA_DIR = './Dataset of Tuberculosis Chest X-rays Images/Dataset of Tuberculosis Chest X-rays Images/Dataset of Tuberculosis Chest X-rays Images'
             class_path = os.path.join(DATA_DIR, cls)
        
        for img_name in os.listdir(class_path):
            image_paths.append(os.path.join(class_path, img_name))
            labels.append(LABEL_MAP['Normal'] if 'Normal' in cls else LABEL_MAP['Tuberculosis'])
except Exception as e:
    print(f"CRITICAL ERROR: Could not find data directory. Error: {e}")
    sys.exit()

df = pd.DataFrame({'path': image_paths, 'label': labels})

# Recreate the exact same 80/10/10 split
train_df, temp_df = train_test_split(df, test_size=0.20, random_state=SEED, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED, stratify=temp_df['label'])

# Use a simple (non-augmented) transform for validation/testing
vis_transform = transforms.Compose([
    transforms.Resize((224, 224)), # Use Resize, not crop, to see the whole image
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# We will run the audit on the TEST set
audit_dataset = TB_Base_Dataset(test_df, transform=vis_transform)
audit_loader = DataLoader(audit_dataset, batch_size=1, shuffle=True) # Batch size 1
print(f"Data loaded. Auditing {len(audit_dataset)} test images.")


# --- 6. Main Audit Loop ---

for model_name, model_path in models_to_audit.items():
    print(f"\n--- Auditing Model: {model_name} ---")
    
    # 1. Create the full model structure
    #    (model_backbone + classifier head)
    model_backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=0)
    num_features = model_backbone.num_features
    model = nn.Sequential(model_backbone, nn.Linear(num_features, 2))

    # -----------------------------------------------------------------
    # <<< *** THIS IS THE CORRECTED LOADING BLOCK *** >>>
    # -----------------------------------------------------------------
    # 2. Load the *fine-tuned* weights
    try:
        # Load the state_dict directly. The fine-tuned .pth files
        # already contain the '0.' and '1.' prefixes from nn.Sequential.
        state_dict = torch.load(model_path, map_location=device)
        
        # Load the weights into our new nn.Sequential model.
        # strict=True ensures all keys match perfectly.
        model.load_state_dict(state_dict, strict=True)
        
        print(f"Successfully loaded fine-tuned weights from {model_path}")
    
    except Exception as e:
        print(f"Error loading weights for {model_name}: {e}")
        print("This is likely a key mismatch. Make sure your .pth file paths are correct.")
        continue # Skip to the next model
    # -----------------------------------------------------------------

    model = model.to(device)
    model.eval()

    # 3. Setup Grad-CAM
    # The target layer is the final normalization layer of the *backbone*
    # model[0] is the 'model_backbone' part of our nn.Sequential
    target_layer = model[0].norm
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    # The target is the "Tuberculosis" class (index 1)
    targets = [ClassifierOutputTarget(1)]

    # 4. Run loop
    print(f"Generating {NUM_IMAGES_TO_AUDIT} sample heatmaps...")
    
    # REMOVED the `with torch.no_grad():` to fix the gradient error
    
    img_count = 0
    for i, (img_tensor, label, img_path) in enumerate(audit_loader):
        if img_count >= NUM_IMAGES_TO_AUDIT:
            break
            
        # We want to see why it made its prediction
        # We'll focus on images that are ACTUALLY Tuberculosis
        if label.item() == 0: # If it's a "Normal" image, skip it
            continue

        print(f"  Processing TB image (Label: {CLASSES[label.item()]})...")
        img_tensor = img_tensor.to(device)
        
        # --- Generate the CAM ---
        grayscale_cam = cam(input_tensor=img_tensor,
                            targets=targets,
                            aug_smooth=False,
                            eigen_smooth=False)
        
        grayscale_cam = grayscale_cam[0, :] # Get the heatmap
        
        # --- Create the Visualization ---
        original_rgb_img = tensor_to_rgb_image(img_tensor[0])
        cam_image = show_cam_on_image(original_rgb_img, grayscale_cam, use_rgb=True)
        
        # --- Plot and Save ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle(f"XAI Audit: {model_name} - Class: TB (Actual)", fontsize=16)

        ax1.imshow(original_rgb_img)
        ax1.set_title("Original Image")
        ax1.axis('off')

        ax2.imshow(cam_image)
        ax2.set_title("Grad-CAM Heatmap (Why it thinks 'TB')")
        ax2.axis('off')
        
        # Use a unique filename
        img_basename = os.path.basename(img_path[0])
        save_name = f"{model_name}__{img_basename}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, save_name))
        plt.close(fig)
        img_count += 1 # Increment our counter for TB images

print("\n--- XAI Final Audit Finished ---")
print(f"All audit images are saved in the '{OUTPUT_DIR}' folder.")