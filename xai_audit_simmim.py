"""Grad-CAM audit entrypoint for the SimMIM-initialized Swin model."""

import argparse

from xai_gradcam_utils import GradCAMAuditor, make_swin_reshape_transform


DEFAULT_MODEL_PATH = "./Pretraining/simmim_swin_backbone.pth"
DEFAULT_DATA_DIR = "./Dataset of Tuberculosis Chest X-rays Images"
DEFAULT_OUTPUT_DIR = "./gradcam_outputs/simmim"


def resolve_target_layer(model):
    return model[0].norm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Grad-CAM on the SimMIM initialized Swin Transformer.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to the fine-tuned SimMIM checkpoint (.pth).")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Root directory containing the TB dataset.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to store Grad-CAM visualizations.")
    parser.add_argument("--num-images", type=int, default=10, help="Number of Tuberculosis samples to visualize.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for the audit DataLoader.")
    parser.add_argument("--image-size", type=int, default=224, help="Square image resolution used for evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset splits and shuffling.")
    parser.add_argument("--aug-smooth", action="store_true", help="Enable augmentation smoothing in Grad-CAM.")
    parser.add_argument("--eigen-smooth", action="store_true", help="Enable eigenvalue smoothing in Grad-CAM.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cam_kwargs = {"aug_smooth": args.aug_smooth, "eigen_smooth": args.eigen_smooth}
    reshape_transform = make_swin_reshape_transform(stage_height=7, stage_width=7)

    auditor = GradCAMAuditor(
        model_name="SimMIM",
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_images=args.num_images,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
        target_layer_resolver=resolve_target_layer,
        cam_kwargs=cam_kwargs,
        reshape_transform=reshape_transform,
    )

    print(f"Running Grad-CAM for {args.num_images} Tuberculosis samples...")
    saved_paths = auditor.run()

    if saved_paths:
        print("Saved Grad-CAM visualizations:")
        for path in saved_paths:
            print(f" - {path}")
    else:
        print("No Tuberculosis samples were visualized. Consider increasing --num-images or checking the dataset split.")


if __name__ == "__main__":
    main()
