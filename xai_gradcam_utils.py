import os
import random
import warnings
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

try:
    from pytorch_grad_cam.utils.reshape_transforms import swin_transform as _lib_swin_transform
except ImportError:  # Older grad-cam releases
    _lib_swin_transform = None


def _fallback_swin_transform(tensor, height, width):
    """Robust Swin reshape that tolerates extra spatial dims."""

    if tensor.ndim < 3:
        raise ValueError(
            "Swin reshape requires at least 3 dimensions; "
            f"got shape {tuple(tensor.size())}."
        )

    if tensor.ndim == 4 and tensor.shape[1] == height and tensor.shape[2] == width:
        # Already spatial with channel-last
        return tensor.permute(0, 3, 1, 2)

    b = tensor.shape[0]
    channels = tensor.shape[-1]
    seq_len = tensor.numel() // (b * channels)

    expected = height * width
    if seq_len != expected:
        raise ValueError(
            "Swin reshape mismatch: expected sequence length "
            f"{expected}, got {seq_len}."
        )

    reshaped = tensor.reshape(b, height, width, channels)
    return reshaped.permute(0, 3, 1, 2)


if _lib_swin_transform is not None:
    def swin_transform(tensor, height, width):
        try:
            return _lib_swin_transform(tensor, height, width)
        except ValueError:
            return _fallback_swin_transform(tensor, height, width)
else:
    swin_transform = _fallback_swin_transform
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm

__all__ = [
    "GradCAMAuditor",
    "resolve_data_dir",
    "make_swin_reshape_transform",
]

TB_CLASS_DIRS = ["Normal Chest X-rays", "TB Chest X-rays"]
DEFAULT_DATA_DIR = "./Dataset of Tuberculosis Chest X-rays Images"
NESTED_FOLDER_NAME = "Dataset of Tuberculosis Chest X-rays Images"
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_data_dir(base_dir: str = DEFAULT_DATA_DIR) -> Path:
    """Return the directory containing the TB class folders.

    The dataset archive is nested with repeated folder names, so we iteratively
    append the folder name until we discover the level that contains the class
    directories. Raises FileNotFoundError if no valid path is detected."""

    candidate = Path(base_dir)
    seen: List[Path] = []

    def has_class_dirs(path: Path) -> bool:
        return path.exists() and all((path / cls).is_dir() for cls in TB_CLASS_DIRS)

    for _ in range(4):  # check base plus three nested levels
        if has_class_dirs(candidate):
            return candidate
        seen.append(candidate)
        candidate = candidate / NESTED_FOLDER_NAME

    searched = "\n".join(str(p) for p in seen)
    raise FileNotFoundError(
        "Unable to locate dataset directory containing class folders. "
        f"Checked the following paths:\n{searched}\n"
        "Please verify the --data-dir argument."
    )


class TBAuditDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Optional[Callable] = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        tensor = self.transform(img) if self.transform else transforms.ToTensor()(img)
        label = torch.tensor(row["label"], dtype=torch.long)
        return tensor, label, row["path"]


def build_tb_dataframe(data_dir: str, seed: int = 42) -> pd.DataFrame:
    resolved = resolve_data_dir(data_dir)
    image_paths: List[str] = []
    labels: List[int] = []
    label_map = {cls: idx for idx, cls in enumerate(TB_CLASS_DIRS)}

    for cls in TB_CLASS_DIRS:
        class_path = resolved / cls
        for fname in os.listdir(class_path):
            img_path = class_path / fname
            if img_path.is_file():
                image_paths.append(str(img_path))
                labels.append(label_map[cls])

    if not image_paths:
        raise RuntimeError(
            f"No images found under {resolved}. Ensure the dataset is extracted correctly."
        )

    df = pd.DataFrame({"path": image_paths, "label": labels})
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def train_val_test_split(df: pd.DataFrame, seed: int = 42):
    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=seed, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed, stratify=temp_df["label"])
    return train_df, val_df, test_df


def build_audit_loader(test_df: pd.DataFrame, image_size: int = 224, batch_size: int = 1) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
    ])
    dataset = TBAuditDataset(test_df, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def build_swin_classifier(num_classes: int = 2) -> nn.Module:
    backbone = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=0)
    head = nn.Linear(backbone.num_features, num_classes)
    return nn.Sequential(backbone, head)


def tensor_to_rgb_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu()
    mean = torch.tensor(DEFAULT_MEAN).view(3, 1, 1)
    std = torch.tensor(DEFAULT_STD).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = tensor.clamp(0, 1)
    return tensor.permute(1, 2, 0).numpy()


def make_swin_reshape_transform(
    *,
    stage_height: int,
    stage_width: int,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the official reshape helper for Swin feature maps."""

    return partial(swin_transform, height=stage_height, width=stage_width)


class GradCAMAuditor:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        data_dir: str,
        output_dir: str,
        *,
        num_images: int = 10,
        batch_size: int = 1,
        image_size: int = 224,
        seed: int = 42,
        target_class_idx: int = 1,
        target_layer_resolver: Optional[Callable[[nn.Module], nn.Module]] = None,
        cam_kwargs: Optional[dict] = None,
        reshape_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.num_images = num_images
        self.batch_size = batch_size
        self.image_size = image_size
        self.seed = seed
        self.target_class_idx = target_class_idx
        self.target_layer_resolver = target_layer_resolver or self.default_target_layer
        self.cam_kwargs = cam_kwargs or {}
        self.reshape_transform = reshape_transform

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.audit_loader: Optional[DataLoader] = None
        self.test_df: Optional[pd.DataFrame] = None

        set_global_seed(seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def default_target_layer(model: nn.Module) -> nn.Module:
        backbone = model[0]
        return backbone.layers[-1].blocks[-1].norm2

    def prepare_data(self) -> None:
        df = build_tb_dataframe(self.data_dir, seed=self.seed)
        _, _, test_df = train_val_test_split(df, seed=self.seed)
        self.test_df = test_df
        self.audit_loader = build_audit_loader(test_df, image_size=self.image_size, batch_size=self.batch_size)

    def prepare_model(self) -> None:
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model weights not found at {self.model_path}")
        model = build_swin_classifier(num_classes=2)
        state_dict = torch.load(self.model_path, map_location=self.device)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            translated = {}
            for key, value in state_dict.items():
                new_key = key.replace("layers_", "layers.") if "layers_" in key else key
                translated[new_key] = value
            model.load_state_dict(translated, strict=False)

        self.model = model.to(self.device).eval()

    def run(self) -> List[Path]:
        if self.model is None:
            self.prepare_model()
        if self.audit_loader is None:
            self.prepare_data()

        assert self.model is not None and self.audit_loader is not None

        target_layer = self.target_layer_resolver(self.model)
        cam = GradCAM(
            model=self.model,
            target_layers=[target_layer],
            reshape_transform=self.reshape_transform,
        )
        targets = [ClassifierOutputTarget(self.target_class_idx)]

        saved_paths: List[Path] = []
        processed = 0

        for img_tensor, label, img_path in self.audit_loader:
            if processed >= self.num_images:
                break
            if label.item() != self.target_class_idx:
                continue

            img_tensor = img_tensor.to(self.device)
            try:
                grayscale_cam = cam(
                    input_tensor=img_tensor,
                    targets=targets,
                    aug_smooth=self.cam_kwargs.get("aug_smooth", False),
                    eigen_smooth=self.cam_kwargs.get("eigen_smooth", False),
                )[0]
            except ValueError as err:
                if "Invalid grads shape" not in str(err):
                    raise
                grayscale_cam = self._generate_cam_manual(img_tensor, targets, target_layer)

            rgb = tensor_to_rgb_image(img_tensor[0])
            cam_image = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle(f"{self.model_name} Grad-CAM", fontsize=16)

            axes[0].imshow(rgb)
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(cam_image)
            axes[1].set_title("Grad-CAM Heatmap")
            axes[1].axis("off")

            img_path_str = img_path[0] if isinstance(img_path, (list, tuple)) else img_path
            filename = f"{self.model_name}__{Path(img_path_str).name}"
            save_path = self.output_dir / filename
            fig.savefig(save_path, bbox_inches="tight")
            plt.close(fig)

            saved_paths.append(save_path)
            processed += 1

        if not saved_paths:
            warnings.warn(
                "No Tuberculosis images were processed. Try increasing the number of sampled "
                "images or ensure the dataset contains TB cases."
            )

        return saved_paths

    def _generate_cam_manual(
        self,
        img_tensor: torch.Tensor,
        targets: List[ClassifierOutputTarget],
        target_layer: nn.Module,
    ) -> np.ndarray:
        activations: dict = {}
        gradients: dict = {}

        def forward_hook(_, __, output):
            activations["value"] = output.detach()

        def backward_hook(_, grad_input, grad_output):
            gradients["value"] = grad_output[0].detach()

        handle_f = target_layer.register_forward_hook(forward_hook)
        handle_b = target_layer.register_full_backward_hook(backward_hook)

        try:
            scores = self.model(img_tensor)
            loss = torch.stack([target(scores) for target in targets]).sum()
            self.model.zero_grad()
            loss.backward(retain_graph=False)
        finally:
            handle_f.remove()
            handle_b.remove()

        acts = activations["value"]
        grads = gradients["value"]

        reshape = self.reshape_transform
        if reshape is None:
            # Fallback: assume final stage of swin_base (7x7)
            reshape = make_swin_reshape_transform(stage_height=7, stage_width=7)

        def ensure_spatial(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.ndim == 4:
                return tensor
            return reshape(tensor)

        acts = ensure_spatial(acts)
        grads = ensure_spatial(grads)

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam
