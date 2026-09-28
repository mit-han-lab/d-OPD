"""Method parameters; no experiment or training recipe."""
from dataclasses import dataclass
from omegaconf import OmegaConf


@dataclass
class MethodConfig:
    block_size: int = 4
    denoise_step: int | None = None
    correction_strength: float = 1.0
    top_k: int = 16
    correctness_gate: bool = True
    loss_top_k: int | None = None
    require_visible_future: bool | None = None
    preserve_zero_support: bool | None = None
    exclude_eos_from_loss: bool | None = None
    score_chunk_size: int | None = None
    cache_max_bytes: int | None = None

    def __post_init__(self):
        required = ("block_size", "top_k", "correctness_gate", "correction_strength")
        missing = [name for name in required if getattr(self, name) is None]
        if missing:
            raise ValueError("Set these method parameters explicitly: " + ", ".join(missing))
        if not isinstance(self.correctness_gate, bool):
            raise ValueError("correctness_gate must be a boolean")
        if self.denoise_step is None:
            self.denoise_step = self.block_size
        for name in ("block_size", "denoise_step", "top_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("require_visible_future", "preserve_zero_support", "exclude_eos_from_loss"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean or null")
        for name in ("score_chunk_size", "cache_max_bytes", "loss_top_k"):
            value = getattr(self, name)
            minimum = 0 if name == "loss_top_k" else 1
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < minimum):
                raise ValueError(f"{name} must be an integer >= {minimum} or null")
        if self.block_size < 2:
            raise ValueError("Future-aware correction requires block_size >= 2")
        if self.correction_strength < 0:
            raise ValueError("correction_strength must be nonnegative")

    def internal(self):
        return OmegaConf.create({"training": {
            "block_size": self.block_size,
            "hindsight_lambda": self.correction_strength,
            "hindsight_top_k": self.top_k,
            "top_k_logits": self.top_k if self.loss_top_k is None else self.loss_top_k,
            "hindsight_correctness_only": self.correctness_gate,
            "exclude_im_end": True if self.exclude_eos_from_loss is None else self.exclude_eos_from_loss,
            "hindsight_require_visible_future": self.require_visible_future,
            "hindsight_preserve_zero_support": self.preserve_zero_support,
            "hindsight_score_chunk_size": self.score_chunk_size,
            "hindsight_cache_max_bytes": self.cache_max_bytes,

        }})


def load_config(path):
    return MethodConfig(**OmegaConf.to_container(OmegaConf.load(path), resolve=True))
