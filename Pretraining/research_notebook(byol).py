import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
import random
import numpy as np
import copy # We'll need this for the model setup

# --- 1. Setup GPU and Reproducibility ---

# Set a random seed for reproducibility
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
    print("Using CPU. (Note: Training will be very slow.)")


# --- 2. Component 1: The Data Pipeline (Augmentations) ---

# Define the set of strong augmentations for BYOL
# We use ImageNet mean/std because the timm Swin model expects it
imagenet_mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# We'll create a class for the BYOL augmentations
class BYOLTransform:
    """
    Defines the two-crop augmentation pipeline for BYOL.
    This class will be passed to our dataset.
    """
    def __init__(self, image_size=224, mean_std=imagenet_mean_std):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # Your notebook loaded 3-channel (RGB) images, so we'll do the same
            # If your images are 1-channel, we'd add transforms.Grayscale(num_output_channels=3)
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),

            # ColorJitter: Apply with 80% probability
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)],
                p=0.8
            ),

            # Grayscale: Apply with 20% probability
            transforms.RandomGrayscale(p=0.2),

            # GaussianBlur: Apply with 50% probability
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))],
                p=0.5
            ),

            transforms.Normalize(mean=mean_std[0], std=mean_std[1])
        ])

    def __call__(self, x):
        # Create two different augmented views of the same image
        view_1 = self.transform(x)
        view_2 = self.transform(x)
        return view_1, view_2

# --- 3. Component 1: The SSL Dataset Class ---

class SSLDataset(Dataset):
    """
    A wrapper dataset for self-supervised learning.
    It takes a standard (image, label) dataset and returns
    (view_1, view_2) for each image, ignoring the labels.
    """
    def __init__(self, underlying_dataset, transform):
        self.underlying_dataset = underlying_dataset
        self.transform = transform

    def __len__(self):
        return len(self.underlying_dataset)

    def __getitem__(self, index):
        # Get the original image (and ignore the label)
        # We assume the underlying dataset returns (image, label)
        image, _ = self.underlying_dataset[index]

        # Apply the BYOL transform to get two views
        view_1, view_2 = self.transform(image)

        return view_1, view_2

print("\n--- Setup Complete ---")
print("BYOLTransform and SSLDataset classes are defined.")

import os
import pandas as pd
import time
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder # Need this for one step

local_dir = '/content/local_dataset'

DATA_DIR = './Dataset of Tuberculosis Chest X-rays Images/Dataset of Tuberculosis Chest X-rays Images/Dataset of Tuberculosis Chest X-rays Images/'

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

class TB_SSL_Base_Dataset(Dataset):
    """
    Simple dataset that returns a (PIL.Image, label) tuple
    based on the DataFrame of file paths.
    """
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the image path and label from the DataFrame
        img_path = self.df.iloc[idx]['path']
        label = self.df.iloc[idx]['label']

        # Load the image, convert to 'RGB' (as you did in your notebook)
        image = Image.open(img_path).convert('RGB')

        # We return the PIL Image itself, our SSL transform will handle the rest
        return image, label

print("TB_SSL_Base_Dataset class defined.")

# 1. Instantiate the base dataset with all 7,208 images
#    (We use the full 'df', not train_df, for SSL pre-training)
full_dataset_pil = TB_SSL_Base_Dataset(df)

# 2. Instantiate the BYOL augmentations we defined in Cell 1
byol_transform = BYOLTransform()

# 3. Create the SSL dataset wrapper
#    This class will now take a PIL image from full_dataset_pil,
#    apply the byol_transform to it twice, and return (view_1, view_2).
ssl_train_dataset = SSLDataset(full_dataset_pil, byol_transform)

# 4. Set Batch Size
#    A Tesla T4 in Colab (16GB VRAM) should handle 32 or 64.
#    If you get a memory error, try lowering this to 32 or 16.
BATCH_SIZE = 32

# 5. Create the DataLoader
ssl_train_loader = DataLoader(ssl_train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=2, # Recommended for Colab
                              pin_memory=True,
                              drop_last=True) # drop_last is important for SSL

print(f"\n--- Data Pipeline Ready ---")
print(f"Created SSL Dataloader with {len(ssl_train_dataset)} images.")
print(f"Batch size: {BATCH_SIZE}")

# You can test the loader by fetching one batch
# (This might take a moment as it applies the heavy augmentations)
# view1_batch, view2_batch = next(iter(ssl_train_loader))
# print(f"\nShape of one batch of view_1: {view1_batch.shape}")
# print(f"Shape of one batch of view_2: {view2_batch.shape}")

import copy # Already imported, but good to be explicit

# --- 2. Component 2: The BYOL Model Architecture ---

class BYOL(nn.Module):
    """
    Implementation of BYOL (Bootstrap Your Own Latent).

    Contains:
        - Student (Online) Network: Encoder, Projector, Predictor
        - Teacher (Target) Network: Encoder, Projector
        - Momentum update logic
    """
    def __init__(self,
                 model_name='swin_base_patch4_window7_224',
                 projection_size=256,
                 projection_hidden_size=4096,
                 predictor_hidden_size=4096,
                 momentum_tau=0.99):

        super().__init__()

        self.tau = momentum_tau

        # --- 1. Define Student (Online) Network ---

        # Get a Swin model from timm
        # We set pretrained=False. We will train it from scratch with SSL.
        self.student_encoder = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0  # Set num_classes=0 to get features before the head
        )

        # Get the feature dimension from the encoder (e.g., 1024 for swin_base)
        encoder_dim = self.student_encoder.num_features

        # Projector (MLP)
        self.student_projector = nn.Sequential(
            nn.Linear(encoder_dim, projection_hidden_size),
            nn.BatchNorm1d(projection_hidden_size),
            nn.ReLU(),
            nn.Linear(projection_hidden_size, projection_size) # Final embedding
        )

        # Predictor (MLP) - Only the student has this!
        self.student_predictor = nn.Sequential(
            nn.Linear(projection_size, predictor_hidden_size),
            nn.BatchNorm1d(predictor_hidden_size),
            nn.ReLU(),
            nn.Linear(predictor_hidden_size, projection_size) # Final prediction
        )

        # --- 2. Define Teacher (Target) Network ---

        # The teacher is a deep copy of the student's encoder and projector
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        self.teacher_projector = copy.deepcopy(self.student_projector)

        # --- 3. Freeze the Teacher Network ---
        # We must freeze the teacher. It will ONLY be updated
        # by the momentum_update() function, not by the optimizer.
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False
        for param in self.teacher_projector.parameters():
            param.requires_grad = False

        print(f"BYOL Model Created:")
        print(f"  Encoder: {model_name} (Dim: {encoder_dim})")
        print(f"  Projection Size: {projection_size}")
        print(f"  Momentum Tau: {self.tau}")


    @torch.no_grad() # Decorator to disable gradients
    def momentum_update(self):
        """
        Performs the exponential moving average (EMA) update
        for the teacher network.

        teacher_weights = tau * teacher_weights + (1 - tau) * student_weights
        """
        # Update teacher_encoder
        for student_param, teacher_param in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            teacher_param.data = self.tau * teacher_param.data + (1.0 - self.tau) * student_param.data

        # Update teacher_projector
        for student_param, teacher_param in zip(self.student_projector.parameters(), self.teacher_projector.parameters()):
            teacher_param.data = self.tau * teacher_param.data + (1.0 - self.tau) * student_param.data

    def forward(self, x1, x2):
        """
        Takes two augmented views (x1, x2) and performs the
        forward pass for both student and teacher.
        """

        # --- Student Pass ---
        # Pass view 1 through the student
        student_features_1 = self.student_encoder(x1)
        student_proj_1 = self.student_projector(student_features_1)
        student_pred_1 = self.student_predictor(student_proj_1)

        # Pass view 2 through the student
        student_features_2 = self.student_encoder(x2)
        student_proj_2 = self.student_projector(student_features_2)
        student_pred_2 = self.student_predictor(student_proj_2)

        # --- Teacher Pass (with gradients disabled) ---
        with torch.no_grad():
            # Pass view 1 through the teacher
            teacher_features_1 = self.teacher_encoder(x1)
            teacher_proj_1 = self.teacher_projector(teacher_features_1)

            # Pass view 2 through the teacher
            teacher_features_2 = self.teacher_encoder(x2)
            teacher_proj_2 = self.teacher_projector(teacher_features_2)

        # We return the student's *predictions* and the teacher's *projections*
        # The loss will compare:
        #   - student_pred_1 with teacher_proj_2
        #   - student_pred_2 with teacher_proj_1
        return student_pred_1, student_pred_2, teacher_proj_1, teacher_proj_2

# --- 3. Component 3: The Loss Function ---

def byol_loss_fn(student_pred, teacher_proj):
    """
    Calculates the BYOL loss.
    The loss is the Mean Squared Error (MSE) between the
    L2-normalized student prediction and teacher projection.

    This is equivalent to: 2 - 2 * (pred * proj).sum()
    """
    # L2 normalize the vectors
    student_pred_norm = F.normalize(student_pred, dim=1)
    teacher_proj_norm = F.normalize(teacher_proj.detach(), dim=1) # Stop gradient to teacher

    # Calculate MSE
    loss = F.mse_loss(student_pred_norm, teacher_proj_norm, reduction='mean')

    # The paper's loss is 2 - 2 * (pred * proj).sum().
    # MSE on L2-normed vectors is equivalent and simpler.
    return loss

print("BYOL loss function 'byol_loss_fn' defined.")


# --- 4. Component 4 (Part 1): Initialize the Model ---

# Instantiate the BYOL model
byol_model = BYOL(
    model_name='swin_base_patch4_window7_224',
    momentum_tau=0.99  # Start with 0.99, we'll update this later
).to(device)

# You can print the model to verify, but it will be very long
# print(byol_model)

import math
from tqdm.auto import tqdm

# --- 4. Component 4 (Part 2): The Training Loop ---

# --- 1. Hyperparameters ---
NUM_EPOCHS = 50   # Start with 50-100 to see. Real training can take 300+
LEARNING_RATE = 1e-4
MOMENTUM_BASE_TAU = 0.99 # The starting value for tau
SAVE_PATH = "./byol_swin_backbone.pth" # Save model to Drive

# --- 2. Optimizer ---
# CRITICAL: The optimizer only gets the STUDENT's parameters.
# The teacher is updated by our momentum rule, not by an optimizer.
optimizer = torch.optim.AdamW([
    {'params': byol_model.student_encoder.parameters()},
    {'params': byol_model.student_projector.parameters()},
    {'params': byol_model.student_predictor.parameters()}
], lr=LEARNING_RATE)

# --- 3. Momentum (tau) Scheduler ---
# This is a trick from the BYOL paper. We slowly increase
# tau from MOMENTUM_BASE_TAU to 1.0 over the course of training.
total_steps = len(ssl_train_loader) * NUM_EPOCHS

def update_momentum(model, step, total_steps, base_tau=MOMENTUM_BASE_TAU):
    """Cosine annealing scheduler for momentum tau."""
    new_tau = 1.0 - (1.0 - base_tau) * (math.cos(math.pi * step / total_steps) + 1) / 2
    model.tau = new_tau

print("--- Starting BYOL Pre-training ---")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Saving model to: {SAVE_PATH}")

# Put the model in training mode
byol_model.train()

global_step = 0
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0

    # Use tqdm for a progress bar
    progress_bar = tqdm(ssl_train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)

    for (view_1, view_2) in progress_bar:
        # Move data to the GPU
        view_1 = view_1.to(device)
        view_2 = view_2.to(device)

        # --- 4. Forward pass ---
        # Get student predictions and teacher projections
        pred_1, pred_2, proj_1, proj_2 = byol_model(view_1, view_2)

        # --- 5. Calculate loss (symmetric loss) ---
        loss_1_to_2 = byol_loss_fn(pred_1, proj_2)
        loss_2_to_1 = byol_loss_fn(pred_2, proj_1)
        loss = loss_1_to_2 + loss_2_to_1

        # --- 6. Backpropagation (updates student) ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- 7. Momentum Update (updates teacher) ---
        byol_model.momentum_update()

        # --- 8. Update momentum tau ---
        update_momentum(byol_model, global_step, total_steps)

        # --- 9. Logging ---
        epoch_loss += loss.item()
        global_step += 1
        progress_bar.set_postfix({'loss': loss.item(), 'tau': byol_model.tau})

    # End of epoch
    avg_epoch_loss = epoch_loss / len(ssl_train_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Avg Loss: {avg_epoch_loss:.4f} - Final Tau: {byol_model.tau:.4f}")

print("--- Training Finished ---")

# --- 5. Save the final backbone ---
# We only care about the student's *encoder* (the backbone)
# We can extract its state_dict and save it.
backbone_state_dict = byol_model.student_encoder.state_dict()
torch.save(backbone_state_dict, SAVE_PATH)

print(f"\nSuccessfully saved the pre-trained Swin backbone to:")
print(SAVE_PATH)
