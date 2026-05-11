import argparse

from vlm.configs.training_configs import TRAINING_CONFIGS, get_training_config
from vlm.training.rl import train_rl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RL training.")

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="receipt-base",
        choices=sorted(TRAINING_CONFIGS),
        help="Run config name.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_training_config(args.config)

    print(f"running RL config: {cfg.name}")
    print(f"loading SFT checkpoint from: {cfg.sft_best_checkpoint}")

    train_rl(cfg)


if __name__ == "__main__":
    main()