import argparse

from vlm.configs.training_configs import TRAINING_CONFIGS, get_training_config
from vlm.training.sft import train_sft
from vlm.training.rl import train_rl
from vlm.evaluation.evaluate import compare_sft_and_rl, evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RL training.")

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="base",
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

    print(f"running config: {cfg.name}")
    # run sft training
    print(f"running SFT training for config: {cfg.name}")
    train_sft(cfg, args.resume)
    # run rl training
    print(f"running RL training for config: {cfg.name}")
    print(f"loading SFT checkpoint from: {cfg.sft_best_checkpoint}")
    train_rl(cfg, args.resume)
    # run evaluation
    print(f"evaluating SFT and RL checkpoints for config: {cfg.name}")
    compare_sft_and_rl(
        cfg=cfg,
        num_samples=None,
    )



if __name__ == "__main__":
    main()