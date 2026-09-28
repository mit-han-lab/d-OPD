"""Launch with torchrun --standalone --nproc_per_node=8 train.py --config configs/train.yaml."""
import argparse
import json
import math
import os
import random
from pathlib import Path
from omegaconf import OmegaConf
from dopd.config import load_config, training_config
from dopd.data import load_records, load_correctness_fn


def main():
    parser = argparse.ArgumentParser(description='Train d-OPD on user-supplied data')
    parser.add_argument('--config', default='configs/train.yaml')
    parser.add_argument('--check-config', action='store_true', help='Validate and print config without loading models')
    parser.add_argument('overrides', nargs='*', help='Overrides such as block_size=8 training.lambda=1.0')
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.check_config:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return
    if cfg.training.correctness_gate and cfg.data.correctness_fn is None:
        raise ValueError('Set data.correctness_fn=your_module:check, or explicitly disable training.correctness_gate')
    check = load_correctness_fn(cfg.data.correctness_fn)
    records = load_records(cfg.data.path)
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    chunk_size = cfg.rollout.prompts_per_step
    effective_batch = world * cfg.training.batch_size_per_gpu * cfg.training.gradient_accumulation_steps
    if chunk_size % effective_batch:
        raise ValueError('rollout.prompts_per_step must be divisible by the global effective batch size')
    if len(records) < chunk_size:
        raise ValueError('Training data must contain at least rollout.prompts_per_step records')
    out = Path(cfg.output_dir).resolve()
    if (out / 'ckpt').exists():
        raise FileExistsError(f'{out} already contains checkpoints; use a new output_dir')
    total_steps = math.ceil(len(records) / chunk_size) * cfg.data.epochs
    config = training_config(cfg, total_steps)
    # Validate the model adapter before constructing distributed training state.
    for model_path in (cfg.model.student, cfg.model.teacher):
        if not Path(model_path).is_dir():
            raise ValueError(f'Use a local model directory: {model_path}')
    student_config = json.loads((Path(cfg.model.student) / 'config.json').read_text())
    teacher_config = json.loads((Path(cfg.model.teacher) / 'config.json').read_text())
    if student_config.get('model_type') != 'a2d-qwen3' or teacher_config.get('model_type') != 'qwen3':
        raise ValueError('This minimal release supports dense a2d-qwen3 students and Qwen3 AR teachers')
    if student_config['vocab_size'] != teacher_config['vocab_size']:
        raise ValueError('Student and teacher must share the same vocabulary and token IDs')
    os.environ['DOPD_SEED'] = str(cfg.seed)
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    if cfg.wandb.enabled:
        import wandb
        # Rank zero owns logging. Config must not contain private experiment IDs.
        if rank == 0:
            config.wandb.run_id = cfg.wandb.run_id or wandb.util.generate_id()
            os.environ['WANDB_NAME'] = cfg.wandb.run_name
    import torch
    import torch.distributed as dist
    from accelerate.utils import set_seed
    from dopd.trainer import init_training, train_one_step
    from dopd.rollout import init_engine, generate
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    set_seed(cfg.seed)
    state = init_training(config)
    engine, tokenizer = init_engine(cfg)
    if rank == 0:
        (out / 'temp_data').mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, out / 'public_config.yaml')
    dist.barrier()
    data_rng = random.Random(cfg.seed)
    step = 0
    for epoch in range(cfg.data.epochs):
        data_rng.shuffle(records)
        for offset in range(0, len(records), chunk_size):
            step += 1
            chunk = records[offset:offset + chunk_size]
            # Pad the final batch from this epoch's start to keep global batch
            # shape consistent across ranks (at most one partial batch/epoch).
            chunk += records[:chunk_size - len(chunk)]
            config.experiment.current_epoch = step
            os.environ['DOPD_SEED'] = str(cfg.seed + step)
            set_seed(cfg.seed + step)
            if rank == 0:
                print(f'd-OPD step {step}/{total_steps}: rollout', flush=True)
            rows = generate(engine, tokenizer, chunk, cfg, step, check)
            if rank == 0:
                with (out / 'temp_data/rollouts.json').open('w') as handle:
                    json.dump(rows, handle, ensure_ascii=False)
            dist.barrier()
            engine.sleep()
            torch.cuda.empty_cache()
            train_one_step(state, config)
            dist.barrier()
            if step < total_steps:
                engine.wake_up_from_path(str(out / 'ckpt/optimized'))
    state['accelerator'].end_training()
    if rank == 0:
        print(f'Finished {total_steps} steps; final checkpoint: {out / "ckpt/optimized"}', flush=True)


if __name__ == '__main__':
    main()
