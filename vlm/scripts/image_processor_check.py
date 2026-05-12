"""
Visualise original CORD image vs what the vision encoder actually sees
after the DonutProcessor resize.

Usage:
    python -m vlm.scripts.viz_image
    python -m vlm.scripts.viz_image --index 5 --config base
    python -m vlm.scripts.viz_image --index 5 --config base --save

The script shows:
- Original image (raw from CORD)
- Processed image (resized to config dimensions — what Donut sees)
- Printed stats: original size, target size, aspect ratios, pixel value stats
"""

import argparse

import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import DonutProcessor

from vlm.configs.training_configs import get_training_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise original vs processed CORD image")
    parser.add_argument("--index", type=int, default=100, help="Dataset sample index")
    parser.add_argument("--config", "-c", default="base", help="Training config name")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--save", action="store_true", help="Save figure to viz_image.png")
    args = parser.parse_args()

    cfg = get_training_config(args.config)

    print(f"loading dataset split={args.split}...")
    ds = load_dataset(cfg.data.dataset_name, split=args.split)
    sample = ds[args.index]
    original = sample["image"].convert("RGB")

    print(f"\nconfig: height={cfg.vision.image_height}, width={cfg.vision.image_width}")
    print(f"config aspect ratio (w/h): {cfg.vision.image_width / cfg.vision.image_height:.3f}")
    print(f"config aspect ratio (h/w): {cfg.vision.image_height / cfg.vision.image_width:.3f}")
    print(f"\noriginal image: {original.width}x{original.height} (w x h)")
    print(f"original aspect ratio (w/h): {original.width / original.height:.3f}")
    print(f"original aspect ratio (h/w): {original.height / original.width:.3f}")
    print(f"orientation: {'portrait (h>w)' if original.height > original.width else 'landscape (w>h)'}")

    # Set up processor with config dimensions.
    processor = DonutProcessor.from_pretrained(cfg.vision.model_name)
    if not cfg.vision.default_processor:
        processor.image_processor.size = {
            "height": cfg.vision.image_height,
            "width": cfg.vision.image_width,
        }

    # Process the image.
    processed = processor(original, return_tensors="pt")
    pixel_values = processed.pixel_values  # (1, 3, H, W)

    print(f"\nprocessed tensor shape: {tuple(pixel_values.shape)}")
    print(f"pixel value range: [{pixel_values.min():.3f}, {pixel_values.max():.3f}]")
    print(f"pixel value mean: {pixel_values.mean():.3f}")

    # Convert tensor back to displayable image.
    img_np = pixel_values[0].permute(1, 2, 0).numpy()
    # Normalise to [0, 1] for display.
    img_min, img_max = img_np.min(), img_np.max()
    img_norm = (img_np - img_min) / (img_max - img_min + 1e-8)

    processed_h, processed_w = img_np.shape[:2]
    print(f"processed image: {processed_w}x{processed_h} (w x h)")
    print(f"processed aspect ratio (w/h): {processed_w / processed_h:.3f}")

    # Check aspect ratio mismatch.
    orig_ratio = original.width / original.height
    proc_ratio = processed_w / processed_h
    distortion = abs(orig_ratio - proc_ratio) / orig_ratio
    print(f"\naspect ratio distortion: {distortion:.1%}")
    if distortion > 0.15:
        print("⚠  significant distortion — images are being squished/stretched")
        print(f"   original is {'portrait' if original.height > original.width else 'landscape'}, "
              f"config is {'portrait' if cfg.vision.image_height > cfg.vision.image_width else 'landscape'}")
    else:
        print("✓  aspect ratios are close — minimal distortion")

    # Plot.
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    axes[0].imshow(original)
    axes[0].set_title(
        f"Original\n{original.width}×{original.height} (w×h)\n"
        f"aspect (w/h): {original.width / original.height:.2f}",
        fontsize=11,
    )
    axes[0].axis("off")

    axes[1].imshow(img_norm)
    axes[1].set_title(
        f"Processed (what Donut sees)\n{processed_w}×{processed_h} (w×h)\n"
        f"aspect (w/h): {processed_w / processed_h:.2f}  "
        f"{'⚠ squished' if distortion > 0.15 else '✓ ok'}",
        fontsize=11,
    )
    axes[1].axis("off")

    fig.suptitle(
        f"CORD sample {args.index} — config: {args.config}  "
        f"(target {cfg.vision.image_width}×{cfg.vision.image_height} w×h)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    if args.save:
        out_path = "viz_image.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\nsaved to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()