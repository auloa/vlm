from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from vlm.utils.training import assert_only_projector_trainable


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_tokenizer(tokenizer):
    tokenizer.padding_side = "right"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def build_instruction(tokenizer, user_message: str) -> str:
    """
    Wrap a plain user message in the tokenizer's chat template.
    Delegates structural details (role markers, EOS placement,
    assistant generation prompt) to the tokenizer so the prompt
    matches the format the model was trained on.
    The returned string is tokenized downstream with
     `add_special_tokens=True`, which adds BOS — the
    template itself doesn't.
    """
    if not tokenizer.chat_template:
        raise ValueError(
            "Tokenizer has no chat_template. Either use a chat/instruct model "
            "or supply a plain-text instruction directly."
        )

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False,
        add_generation_prompt=True,
    )


def set_projector_only_trainable(model) -> None:
    """Freeze the base models and train only the projector."""
    for param in model.vision_encoder.parameters():
        param.requires_grad_(False)

    for param in model.lm.model.parameters():
        param.requires_grad_(False)

    for param in model.projector.parameters():
        param.requires_grad_(True)

    model.vision_encoder.eval()
    model.lm.model.eval()

    assert_only_projector_trainable(model)


def log_text(
    writer: SummaryWriter,
    tag: str,
    text: str,
    step: int | None = None,
) -> None:
    """Print a message and optionally mirror it to TensorBoard."""
    print(text)
    if step is not None:
        writer.add_text(tag, f"```text\n{text}\n```", step)