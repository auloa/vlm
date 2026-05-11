import argparse
from pathlib import Path

from vlm.configs.training_configs import TRAINING_CONFIGS, get_training_config
from vlm.evaluation.evaluate import compare_sft_and_rl, evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate receipt JSON extraction.")

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="receipt_base",
        choices=sorted(TRAINING_CONFIGS),
        help="Run config name.",
    )

    parser.add_argument(
        "--stage",
        type=str,
        default="both",
        choices=["sft", "rl", "both", "custom"],
        help="Which checkpoint to evaluate.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. Required when --stage custom.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of held-out test examples to evaluate.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_training_config(args.config)

    if args.stage == "both":
        compare_sft_and_rl(
            cfg=cfg,
            num_samples=args.num_samples,
        )
        return

    if args.stage == "sft":
        checkpoint_path = cfg.sft_best_checkpoint
    elif args.stage == "rl":
        checkpoint_path = cfg.rl_best_checkpoint
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when --stage custom")
        checkpoint_path = Path(args.checkpoint)

    evaluate_checkpoint(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        stage=args.stage,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()