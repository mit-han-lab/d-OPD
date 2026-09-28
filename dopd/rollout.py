"""Online student rollouts with the original block diffusion inference backend."""
import random
import torch
import torch.distributed as dist
from dopd.data import format_prompt


def init_engine(cfg):
    from transformers import AutoTokenizer
    from dopd.modeling import _register_a2d_model_classes
    from dopd.mask_token_utils import ensure_diffusion_mask_token
    from dopd.jetengine.llm import LLM
    _register_a2d_model_classes()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.student, trust_remote_code=False)
    ensure_diffusion_mask_token(tokenizer)
    engine = LLM(
        cfg.model.student, tensor_parallel_size=1, enforce_eager=True,
        mask_token_id=tokenizer.mask_token_id, pad_token_id=tokenizer.pad_token_id,
        block_length=cfg.block_size,
        copy_pad_embedding_to_mask=cfg.model.copy_pad_embedding_to_mask,
        gpu_memory_utilization=cfg.rollout.gpu_memory_utilization,
        max_model_len=cfg.rollout.max_prompt_tokens + cfg.rollout.max_tokens,
        max_num_batched_tokens=max(16384, cfg.rollout.max_prompt_tokens + cfg.rollout.max_tokens),
    )
    return engine, tokenizer


def generate(engine, tokenizer, records, cfg, step, correctness_fn):
    from dopd.jetengine.sampling_params import SamplingParams
    rank, world = dist.get_rank(), dist.get_world_size()
    seed = int(cfg.seed) + step
    prompts = [format_prompt(record, tokenizer, cfg) for record in records]
    # Reject overlong prompts before generation, rather than silently misaligning
    # the trajectory map / correctness labels when the trainer drops a row.
    for prompt in prompts:
        if len(tokenizer.encode(prompt)) > cfg.rollout.max_prompt_tokens:
            raise ValueError('Prompt exceeds rollout.max_prompt_tokens; filter data or raise the limit')
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    indices = indices[rank::world]
    params = [SamplingParams(
        block_length=cfg.block_size, denoising_steps=cfg.denoise_step,
        max_tokens=cfg.rollout.max_tokens, temperature=cfg.rollout.temperature,
        topp=cfg.rollout.top_p, topk=cfg.rollout.top_k,
        remasking_strategy=cfg.rollout.remasking_strategy,
        dynamic_threshold=cfg.rollout.dynamic_threshold,
        response_aligned_blocks=cfg.rollout.response_aligned_blocks,
        seed=(seed * 1000003 + index) % (2**31 - 1),
    ) for index in indices]
    outputs = engine.generate_streaming([prompts[i] for i in indices], params,
                                       max_active=cfg.rollout.max_active) if indices else []
    if len(outputs) != len(indices):
        raise RuntimeError('Rollout count does not match input prompts')
    gathered = [None] * world
    dist.all_gather_object(gathered, list(zip(indices, outputs)))
    if rank != 0:
        return None
    rows = []
    for index, output in sorted(item for group in gathered for item in group):
        response = output['text']
        step_map = output['first_unmask_times']
        if not step_map or len(step_map) != len(output['token_ids']):
            raise RuntimeError('Rollout did not return a valid denoising trajectory')
        correct = bool(correctness_fn(records[index], response)) if correctness_fn else False
        rows.append({'prompt': prompts[index], 'response': response,
                     'step_map': step_map, 'reward': float(correct),
                     'hindsight_correctness': correct})
    return rows
