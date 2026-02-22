# --- [NEW] Cell 2 (Combined Imports & Setup) ---

# Core PyTorch and ML Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
import random
import numpy as np
import copy # We will need this

# Your existing file/OS imports
import os
import pandas as pd
import time
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder

# --- Setup GPU and Reproducibility ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Check for GPU and set device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    # These are good for reproducibility on CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    device = torch.device("cpu")
    print("Using CPU.")

# Define ImageNet mean/std (this was also in the deleted cell)
imagenet_mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

import os
import pandas as pd
import time
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder # Need this for one step

# --- Define file paths ---
DATA_DIR = './Dataset of Tuberculosis Chest X-rays Images/Dataset of Tuberculosis Chest X-rays Images/Dataset of Tuberculosis Chest X-rays Images'

# Define the class names (subfolder names from your zip file)
CLASSES = ['Normal Chest X-rays', 'TB Chest X-rays']

# Assign integer labels: Normal: 0, Tuberculosis: 1
LABEL_MAP = {k: v for v, k in enumerate(CLASSES)}

# Create lists to hold file paths and their corresponding labels
image_paths = []
labels = []

print(f"\nLoading file paths from '{DATA_DIR}'...")

# Populate the lists by iterating through the subfolders
for cls in CLASSES:
    class_path = os.path.join(DATA_DIR, cls)
    if not os.path.isdir(class_path):
        print(f"Error: Directory not found at {class_path}")
        continue
    for img_name in os.listdir(class_path):
        image_paths.append(os.path.join(class_path, img_name))
        labels.append(LABEL_MAP[cls])

# Create a DataFrame for better visualization and handling
df = pd.DataFrame({
    'path': image_paths,
    'label': labels
})

if df.empty:
    print("Error: No images found. Please check the DATA_DIR path.")
else:
    # Display the class distribution
    print("\nFull Dataset Class Distribution:")
    print(df['label'].value_counts())
    print(f"\nTotal images found: {len(df)}")

class TB_Base_Dataset(Dataset):
    """
    Flexible dataset that returns a (PIL.Image, label) tuple
    based on the DataFrame of file paths.
    It can now apply an optional transform.
    """
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform # Store the transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the image path and label from the DataFrame
        img_path = self.df.iloc[idx]['path']
        label = self.df.iloc[idx]['label']

        # Load the image, convert to 'RGB'
        image = Image.open(img_path).convert('RGB')

        # Apply the transform if it exists
        if self.transform:
            image = self.transform(image)

        return image, label

print("TB_Base_Dataset class (v2) defined.")

# 1. Define the simple transform for SimMIM
# Note: 'imagenet_mean_std' was already defined in your Cell 2
simmim_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # Convert PIL image to Tensor
    transforms.Normalize(mean=imagenet_mean_std[0], std=imagenet_mean_std[1]) # Normalize Tensor
])

# 2. Instantiate the base dataset with the new SimMIM transform
# We use the full 'df' (all 7,208 images)
full_dataset_simmim = TB_Base_Dataset(df, transform=simmim_transform)

# 3. Set Batch Size
# This model is lighter than BYOL (no teacher network).
# We can start with 64. If we get a CUDA OOM error, we'll drop it to 32.
BATCH_SIZE = 64

# 4. Create the DataLoader
simmim_train_loader = DataLoader(full_dataset_simmim,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True,
                                 num_workers=2, # Recommended for Colab
                                 pin_memory=True,
                                 drop_last=True)

print(f"\n--- SimMIM Data Pipeline Ready ---")
print(f"Created Dataloader with {len(full_dataset_simmim)} images.")
print(f"Batch size: {BATCH_SIZE}")
print("\nNote: The 'BYOLTransform' and 'SSLDataset' from Cell 2 are not used here.")

# --- Component 2: The SimMIM Model Architecture ---

class SimMIM(nn.Module):
    """
    A SimMIM-style implementation for Swin Transformer.

    This model will:
    1. Take a batch of images.
    2. Randomly mask (zero-out) a portion of the patches.
    3. Feed the masked image through the Swin encoder.
    4. Use a simple linear decoder to predict the pixels of *all* patches.
    5. The loss will then be calculated on the *masked* patches only.
    """
    def __init__(self,
                 model_name='swin_base_patch4_window7_224',
                 patch_size=32,
                 img_size=224,
                 mask_ratio=0.75):

        super().__init__()

        self.mask_ratio = mask_ratio

        # Calculate patch dimensions
        assert img_size % patch_size == 0, "Img size must be divisible by patch size."
        self.num_patches_side = img_size // patch_size # 224 // 32 = 7
        self.num_patches_total = self.num_patches_side ** 2 # 7*7 = 49

        self.patch_size = patch_size
        self.patch_dim = 3 * patch_size * patch_size # 3 * 32 * 32 = 3072

        # 1. --- Encoder ---
        # We get the feature maps from the encoder
        # `features_only=True` returns a list of feature maps from each stage
        self.encoder = timm.create_model(
            model_name,
            pretrained=False,
            features_only=True
        )

        # The last feature map from swin_base (stage 4) has 1024 channels
        # and a 7x7 grid, which matches our 49 patches.
        encoder_dim = 1024

        # 2. --- Decoder ---
        # The "Simple" in SimMIM: a single Linear layer.
        # It projects the 1024-dim feature from a patch back to
        # the 3072 raw pixel values for that patch.
        self.decoder = nn.Linear(encoder_dim, self.patch_dim)

        print(f"SimMIM Model Created:")
        print(f"  Encoder: {model_name} (Dim: {encoder_dim})")
        print(f"  Patch Size: {patch_size}x{patch_size}")
        print(f"  Total Patches: {self.num_patches_total} (7x7)")
        print(f"  Masking Ratio: {self.mask_ratio}")

    def patchify(self, imgs):
        """
        Converts a batch of images to a batch of patches.
        Input: (B, 3, 224, 224)
        Output: (B, 49, 3072)
        """
        p = self.patch_size
        h = w = self.num_patches_side
        B, C, H, W = imgs.shape

        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 1, 3, 5) # (B, h, w, C, p, p)
        x = x.reshape(B, h * w, C * p * p) # (B, 49, 3072)
        return x

    def unpatchify(self, x):
        """
        Converts a batch of patches back to a batch of images.
        Input: (B, 49, 3072)
        Output: (B, 3, 224, 224)
        """
        p = self.patch_size
        h = w = self.num_patches_side
        B, N, D = x.shape # (B, 49, 3072)

        x = x.reshape(B, h, w, 3, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5) # (B, 3, h, p, w, p)
        x = x.reshape(B, 3, h * p, w * p) # (B, 3, 224, 224)
        return x

    def generate_mask(self, B):
        """
        Generates a random mask for the patches.
        Output: (B, 49) - 1s for MASKED, 0s for VISIBLE
        """
        num_masked = int(self.mask_ratio * self.num_patches_total)

        # Generate a random permutation of indices
        rand_indices = torch.rand(B, self.num_patches_total, device=device).argsort()
        mask_indices = rand_indices[:, :num_masked] # Indices to mask

        # Create the mask (1s for masked, 0s for visible)
        mask = torch.zeros(B, self.num_patches_total, device=device, dtype=torch.float32)
        mask.scatter_(1, mask_indices, 1)
        return mask

    def forward(self, x):
        # 1. Get original patches (this will be our target)
        # target_patches shape: (B, 49, 3072)
        target_patches = self.patchify(x)

        # 2. Generate the mask
        # mask shape: (B, 49)
        mask = self.generate_mask(x.shape[0])

        # 3. Create the masked image
        # We need to reshape the mask to multiply it with the image
        # (B, 49) -> (B, 49, 1) -> (B, 49, 3072)
        mask_for_patches = mask.unsqueeze(-1).expand_as(target_patches)

        # (1 - mask) keeps visible patches, (mask) zeroes out masked patches
        masked_patches = target_patches * (1.0 - mask_for_patches)

        # x_masked shape: (B, 3, 224, 224)
        x_masked = self.unpatchify(masked_patches)

        # 4. Forward pass through encoder
        # features is a list of 4 feature maps
        # We take the last one: (B, 1024, 7, 7)
        features = self.encoder(x_masked)[-1]

        # 5. Forward pass through decoder
        # (B, 1024, 7, 7) -> (B, 7, 7, 1024) -> (B, 49, 1024)
        features = features.permute(0, 2, 3, 1).reshape(x.shape[0], self.num_patches_total, -1)

        # predicted_patches shape: (B, 49, 3072)
        predicted_patches = self.decoder(features)

        # Return everything needed for the loss
        # prediction, target, mask (1s = masked)
        return predicted_patches, target_patches, mask

# --- Instantiate the Model ---
simmim_model = SimMIM().to(device)

# You can test with a dummy batch
# with torch.no_grad():
#     dummy_batch = torch.randn(2, 3, 224, 224).to(device)
#     pred, target, mask = simmim_model(dummy_batch)
#     print(f"Prediction shape: {pred.shape}") # (2, 49, 3072)
#     print(f"Target shape: {target.shape}")   # (2, 49, 3072)
#     print(f"Mask shape: {mask.shape}")       # (2, 49)

# --- Component 3: The SimMIM Loss Function ---

def simmim_loss_fn(prediction, target, mask):
    """
    Calculates the L1 (Mean Absolute Error) loss only on the masked patches.

    Args:
    - prediction (torch.Tensor): The model's predicted patches.
                                 Shape: (B, num_patches, patch_dim)
    - target (torch.Tensor): The ground truth patches.
                             Shape: (B, num_patches, patch_dim)
    - mask (torch.Tensor): The mask. 1s = MASKED, 0s = VISIBLE.
                           Shape: (B, num_patches)
    """

    # Calculate L1 loss for all patches
    loss_all_patches = F.l1_loss(prediction, target, reduction='none')

    # We need to expand the mask to match the loss shape
    # (B, 49) -> (B, 49, 1)
    mask = mask.unsqueeze(-1)

    # Apply the mask: only keep the loss from masked patches
    loss_masked_patches = loss_all_patches * mask

    # Calculate the mean loss
    # We divide the sum of loss by the *number of masked pixels*
    # (B, 49, 3072) -> (B, 49) -> (B, 1) -> scalar
    num_masked_pixels = mask.sum() * prediction.shape[-1] # (num_masked_patches * patch_dim)

    # Handle potential division by zero if mask is all zero (very unlikely)
    if num_masked_pixels == 0:
        return loss_masked_patches.sum()

    mean_loss = loss_masked_patches.sum() / num_masked_pixels

    return mean_loss

print("SimMIM loss function 'simmim_loss_fn' defined.")

from tqdm.auto import tqdm # For a nice progress bar

# --- 4. Component 4: The Training Loop ---

# --- 1. Hyperparameters ---
NUM_EPOCHS = 100  # Masked models often train a bit faster. 100 is a good start.
LEARNING_RATE = 1e-4
SAVE_PATH = "./simmim_swin_backbone.pth" 

# --- 2. Optimizer ---
# The optimizer gets ALL parameters of the SimMIM model (encoder + decoder)
optimizer = torch.optim.AdamW(simmim_model.parameters(),
                               lr=LEARNING_RATE,
                               weight_decay=0.05)

print("--- Starting SimMIM Pre-training ---")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Saving model to: {SAVE_PATH}")

# Put the model in training mode
simmim_model.train()

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0

    # Use tqdm for a progress bar
    progress_bar = tqdm(simmim_train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)

    # Note: simmim_train_loader returns (image, label). We only need the image.
    for (images, _) in progress_bar:

        # Move images to the GPU
        images = images.to(device)

        # --- 3. Forward pass ---
        # Get the model's output
        predicted_patches, target_patches, mask = simmim_model(images)

        # --- 4. Calculate loss ---
        loss = simmim_loss_fn(predicted_patches, target_patches, mask)

        # --- 5. Backpropagation ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- 6. Logging ---
        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    # End of epoch
    avg_epoch_loss = epoch_loss / len(simmim_train_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Avg Loss: {avg_epoch_loss:.4f}")

print("--- Training Finished ---")

# --- 7. Save the final backbone ---
# We only care about the *encoder* (the backbone)
backbone_state_dict = simmim_model.encoder.state_dict()
torch.save(backbone_state_dict, SAVE_PATH)

print(f"\nSuccessfully saved the pre-trained Swin backbone to:")
print(SAVE_PATH)
