import argparse

from vlm.configs.training_configs import TRAINING_CONFIGS, get_training_config
from vlm.training.sft import train_sft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SFT training.")

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="debug",
        choices=sorted(TRAINING_CONFIGS),
        help="Run config name.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "Resume from the latest epoch checkpoint for this config. "
            "Restores projector weights, optimizer state, and scheduler state. "
            "If no checkpoint exists, starts from scratch."
        ),
    )


    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_training_config(args.config)

    print(f"running SFT configuration: {cfg.name}")
    train_sft(cfg, args.resume)


if __name__ == "__main__":
    main()