from collections.abc import Callable

from vlm.configs.training_schema import TrainingConfig

ConfigFactory = Callable[[], TrainingConfig]
TRAINING_CONFIGS: dict[str, ConfigFactory] = {}


def _run_config_name(fn: Callable) -> str:
    return fn.__name__


def register_config(fn: Callable[[str], TrainingConfig]) -> ConfigFactory:
    """Register a run config using the function name as the config key."""
    name = _run_config_name(fn)

    def wrapped() -> TrainingConfig:
        return fn(name)

    TRAINING_CONFIGS[name] = wrapped
    return wrapped


def _base_receipt_config(name: str) -> TrainingConfig:
    """Base receipt-extraction configuration.

    Reflects best-known settings from ablations:
    - Batch 4,  gradient accumulation step of 4 (affective batch size 16)
    - Dropout 0.15 in projector (prevents late-epoch overfitting)
    - max_target_length 256 (fits CORD-native schema at ~90% sample retention)

    Individual configs only override fields that differ from this base.
    """
    cfg = TrainingConfig(name=name)

    # Data
    cfg.data.dataset_name = "naver-clova-ix/cord-v2"
    cfg.data.train_split = "train"
    cfg.data.val_split = "validation"
    cfg.data.test_split = "test"
    cfg.data.train_samples = 800
    cfg.data.val_samples = 100
    cfg.data.test_samples = 100

    # Vision encoder
    cfg.vision.model_name = "naver-clova-ix/donut-base-finetuned-cord-v2"
    cfg.vision.default_processor = False
    cfg.vision.image_height = 960
    cfg.vision.image_width = 640

    # Language model
    cfg.model.lm_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    cfg.model.instruction = (
        "Extract the tabular data from this document and output it in JSON format."
    )

    # Projector
    cfg.projector.cross_attention = False
    cfg.projector.num_queries = 64
    cfg.projector.num_heads = 8
    cfg.projector.num_layers = 2
    cfg.projector.ffn_mult = 4
    cfg.projector.projector_mult = 2
    cfg.projector.dropout = 0.15

    # SFT
    cfg.sft.epochs = 15
    cfg.sft.batch_size = 4
    cfg.sft.grad_accum_steps = 4  # (effective batch size 16: more stable training with this data size)
    cfg.sft.learning_rate = 5e-5
    cfg.sft.weight_decay = 0.01
    cfg.sft.grad_clip_norm = 0.5
    cfg.sft.max_target_length = 256
    cfg.sft.log_every = 10
    cfg.sft.sample_every = 40

    # RL
    cfg.rl.epochs = 1
    cfg.rl.completions_per_image = 4
    cfg.rl.learning_rate = 5e-6
    cfg.rl.weight_decay = 0.01
    cfg.rl.temperature = 0.7
    cfg.rl.max_completion_tokens = 256
    cfg.rl.grad_clip_norm = 0.5
    cfg.rl.kl_coef = 0.02
    cfg.rl.ema_alpha = 0.05
    cfg.rl.save_every_n_steps = 200
    cfg.rl.max_steps = 500
    cfg.rl.early_stop_patience = 300
    cfg.rl.early_stop_min_delta = 0.001
    cfg.rl.ppo_epochs = 1
    cfg.rl.clip_eps = 0.2
    cfg.rl.log_every = 10
    cfg.rl.sample_every = 40

    # Eval
    cfg.eval.num_samples = 50
    cfg.eval.max_completion_tokens = 256
    cfg.eval.temperature = 0.1

    return cfg


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────

@register_config
def debug(name: str) -> TrainingConfig:
    """Full pipeline on 20 samples. Verifies env in a few minutes."""
    cfg = _base_receipt_config(name)
    cfg.data.train_samples = 20
    cfg.data.val_samples = 10
    cfg.data.test_samples = 10
    cfg.sft.epochs = 1
    cfg.sft.batch_size = 1
    cfg.sft.grad_accum_steps = 1
    cfg.sft.log_every = 1
    cfg.sft.sample_every = 5
    cfg.rl.epochs = 1
    cfg.rl.completions_per_image = 2
    cfg.rl.max_steps = 50
    cfg.rl.early_stop_patience = 30
    cfg.rl.log_every = 1
    cfg.rl.sample_every = 5
    cfg.eval.num_samples = 10
    return cfg


# ─────────────────────────────────────────────
# Primary experiments
# ─────────────────────────────────────────────

@register_config
def base(name: str) -> TrainingConfig:
    """Submitted model. Batch 4, dropout 0.15, 15 epochs, KL 0.02.

    Inherits all base defaults unchanged. Use this as the reference point
    for all other experiments.
    """
    cfg = _base_receipt_config(name)
    return cfg

@register_config
def b4_stable_short(name: str) -> TrainingConfig:
    cfg = _base_receipt_config(name)

    cfg.sft.batch_size = 4
    cfg.sft.grad_accum_steps = 1
    cfg.sft.epochs = 15
    cfg.sft.learning_rate = 5e-5
    cfg.sft.weight_decay = 0.01
    cfg.sft.grad_clip_norm = 0.5
    cfg.projector.dropout = None

    cfg.rl.learning_rate = 3e-6
    cfg.rl.kl_coef = 0.10
    cfg.rl.max_steps = 200
    cfg.rl.epochs = 1

    return cfg

def b4_stable_short_drop015(name: str) -> TrainingConfig:
    cfg = _base_receipt_config(name)

    cfg.sft.batch_size = 4
    cfg.sft.grad_accum_steps = 1
    cfg.sft.epochs = 15
    cfg.sft.learning_rate = 5e-5
    cfg.sft.weight_decay = 0.01
    cfg.sft.grad_clip_norm = 0.5
    cfg.projector.dropout = 0.15

    cfg.rl.learning_rate = 3e-6
    cfg.rl.kl_coef = 0.10
    cfg.rl.max_steps = 200
    cfg.rl.epochs = 1

    return cfg


@register_config
def b32_e15_drop_01(name: str) -> TrainingConfig:
    """
    More SFT epochs (15) with dropout.
    """
    cfg = _base_receipt_config(name)
    cfg.sft.batch_size = 4
    cfg.sft.grad_accum_steps = 8
    cfg.sft.epochs = 15
    cfg.sft.learning_rate = 5e-5
    cfg.projector.dropout = 0.1
    cfg.rl.kl_coef = 0.2

    return cfg



def get_training_config(name: str) -> TrainingConfig:
    name = name.replace("-", "_").replace(" ", "_")
    try:
        return TRAINING_CONFIGS[name]()
    except KeyError as exc:
        available = ", ".join(sorted(TRAINING_CONFIGS))
        raise ValueError(f"Unknown config '{name}'. Available: {available}") from exc