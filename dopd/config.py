"""Public configuration and translation to the extracted training implementation."""
from pathlib import Path
from omegaconf import OmegaConf


def load_config(path, overrides=()):
    from dopd.defaults import DEFAULT_CONFIG
    defaults = OmegaConf.create(DEFAULT_CONFIG)
    OmegaConf.set_struct(defaults, True)

    def supplied(value):
        if isinstance(value, dict):
            return {key: supplied(item) for key, item in value.items() if item is not None or key == "denoise_step"}
        return value

    cfg = OmegaConf.merge(defaults, supplied(OmegaConf.to_container(OmegaConf.load(path))))
    method_path = Path(path).with_name("method.yaml")
    if method_path.exists():
        method = OmegaConf.to_container(OmegaConf.load(method_path))
        mapping = {
            "block_size": "block_size", "denoise_step": "denoise_step",
            "top_k": "training.candidate_top_k", "correctness_gate": "training.correctness_gate",
            "correction_strength": "training.lambda", "loss_top_k": "training.loss_top_k",
            "require_visible_future": "training.require_visible_future",
            "preserve_zero_support": "training.preserve_zero_support",
            "exclude_eos_from_loss": "training.exclude_eos_from_loss",
            "score_chunk_size": "training.score_chunk_size", "cache_max_bytes": "training.cache_max_bytes",
        }
        for key, value in method.items():
            if key not in mapping:
                raise ValueError(f"Unknown method parameter: {key}")
            if value is not None:
                OmegaConf.update(cfg, mapping[key], value)
    cfg = OmegaConf.merge(cfg, supplied(OmegaConf.to_container(OmegaConf.from_dotlist(list(overrides)))))
    if cfg.denoise_step is None:
        cfg.denoise_step = cfg.block_size
    for key in ('block_size', 'denoise_step', 'data.epochs', 'rollout.prompts_per_step',
                'rollout.max_tokens', 'rollout.max_prompt_tokens', 'rollout.max_active',
                'training.batch_size_per_gpu', 'training.gradient_accumulation_steps',
                'training.candidate_top_k', 'training.score_chunk_size', 'training.save_every'):
        value = OmegaConf.select(cfg, key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{key} must be a positive integer')
    if cfg.block_size < 2:
        raise ValueError('d-OPD requires block_size >= 2 for future context')
    if cfg.training['lambda'] < 0:
        raise ValueError('training.lambda must be nonnegative')
    if not 0 < cfg.rollout.gpu_memory_utilization < 1:
        raise ValueError('rollout.gpu_memory_utilization must lie in (0, 1)')
    return cfg


def training_config(cfg, total_steps):
    t, r = cfg.training, cfg.rollout
    return OmegaConf.create({
        'experiment': {'project': str(Path(cfg.output_dir).resolve()),
            'start_from_scratch': True, 'current_epoch': 1, 'total_step': total_steps,
            'stop_RL_step': total_steps, 'save_every': t.save_every,
            'epoch_checkpoints_to_keep': t.keep_checkpoints, 'save_final_epoch_checkpoint': True},
        'model': {'pretrained_model': cfg.model.student, 'teacher_model': cfg.model.teacher,
            'hindsight_scorer_model': cfg.model.teacher, 'optimized_name': 'optimized',
            'copy_pad_embedding_to_mask': cfg.model.copy_pad_embedding_to_mask},
        'dataset': {'optimization_data': 'rollouts'},
        'rollout': {'num_task_per_step': r.prompts_per_step, 'num_response_per_task': 1},
        'training': {'seed': cfg.seed, 'block_size': cfg.block_size,
            'hindsight_lambda': t['lambda'], 'hindsight_correctness_only': t.correctness_gate,
            'hindsight_top_k': t.candidate_top_k, 'hindsight_require_visible_future': t.require_visible_future,
            'hindsight_preserve_zero_support': t.preserve_zero_support, 'hindsight_score_chunk_size': t.score_chunk_size,
            'hindsight_cache_max_bytes': t.cache_max_bytes,
            'top_k_logits': t.loss_top_k, 'batch_size_lm': t.batch_size_per_gpu,
            'gradient_accumulation_steps': t.gradient_accumulation_steps,
            'mixed_precision': 'bf16', 'enable_tf32': True, 'num_train_epochs': 1,
            'max_grad_norm': t.max_grad_norm, 'max_prompt_len': r.max_prompt_tokens,
            'max_gen_length': r.max_tokens, 'post_num': 0, 'shrink': 1,
            'lower_p': 0.1, 'upper_p': 0.9, 'exclude_im_end': t.exclude_eos_from_loss,
            'gradient_checkpointing_enable': t.gradient_checkpointing,
            'zero_stage': 3, 'offload_optimizer_device': 'cpu', 'offload_param_device': 'cpu'},
        'optimizer': {'name': 'adamw', 'params': {'learning_rate': t.learning_rate,
            'beta1': 0.9, 'beta2': 0.999, 'weight_decay': 0., 'epsilon': 1e-8}},
        'lr_scheduler': {'scheduler': 'cosine', 'params': {'warmup_steps': t.warmup_steps,
            'min_lr_scale': t.min_lr_scale}},
        'wandb': OmegaConf.to_container(cfg.wandb, resolve=True),
    })

from dopd.method_config import MethodConfig
