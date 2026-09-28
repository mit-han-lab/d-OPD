import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import json
import logging
import math
import random
import shutil
import time
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
import wandb
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from dopd.mask_token_utils import copy_pad_embedding_to_mask, ensure_diffusion_mask_token
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from dopd.prompting_utils import UniversalPrompting
from dopd.lr_schedulers import get_scheduler
from dopd.logging import set_verbosity_info, set_verbosity_error
from torch.utils.data import Dataset, DataLoader
from dopd.utils import flatten_omega_conf, AverageMeter
logger = get_logger(__name__, log_level='INFO')
TRAINING_STATE_TAG = 'student'
_HINDSIGHT_STAT_KEYS = ('rows', 'correct_rows', 'rows_with_jobs', 'rows_with_apply', 'eligible_left_positions', 'jobs', 'applied_positions', 'candidate_tokens', 'finite_candidate_tokens', 'scored_branches', 'skipped_no_visible_blocks', 'skipped_no_future_blocks', 'skipped_no_loss_positions', 'skipped_no_candidates', 'skipped_correctness_only_rows', 'skipped_nonfinite_jobs', 'skipped_bad_normalizer_jobs')
_HINDSIGHT_SUM_METRIC_NAMES = tuple((f'train/hindsight_{key}' for key in _HINDSIGHT_STAT_KEYS))

def _reset_hindsight_stats(stats, rows=0):
    stats.clear()
    for key in _HINDSIGHT_STAT_KEYS:
        stats[key] = 0
    stats['rows'] = int(rows)

def _add_hindsight_metrics(loss_metrics, stats):
    if not stats:
        return
    for key in _HINDSIGHT_STAT_KEYS:
        loss_metrics[f'train/hindsight_{key}'] = (1, float(stats.get(key, 0)))

def _metric_to_float(value):
    return float(value.item()) if hasattr(value, 'item') else float(value)

def _add_hindsight_derived_metrics(logged_metrics, totals=None):
    if 'train/hindsight_rows' not in logged_metrics:
        return
    for key in _HINDSIGHT_SUM_METRIC_NAMES:
        if key in logged_metrics:
            logged_metrics[key] = _metric_to_float(logged_metrics[key])
            if totals is not None:
                totals[key] = totals.get(key, 0.0) + logged_metrics[key]
    rows = logged_metrics.get('train/hindsight_rows', 0.0)
    jobs = logged_metrics.get('train/hindsight_jobs', 0.0)
    applied = logged_metrics.get('train/hindsight_applied_positions', 0.0)
    rows_with_apply = logged_metrics.get('train/hindsight_rows_with_apply', 0.0)
    candidates = logged_metrics.get('train/hindsight_candidate_tokens', 0.0)
    logged_metrics['train/hindsight_did_apply'] = 1.0 if applied > 0 else 0.0
    logged_metrics['train/hindsight_apply_rate'] = applied / jobs if jobs > 0 else 0.0
    logged_metrics['train/hindsight_rows_apply_rate'] = rows_with_apply / rows if rows > 0 else 0.0
    logged_metrics['train/hindsight_jobs_per_row'] = jobs / rows if rows > 0 else 0.0
    logged_metrics['train/hindsight_candidates_per_job'] = candidates / jobs if jobs > 0 else 0.0

def _log_hindsight_summary(accelerator, totals):
    rows = totals.get('train/hindsight_rows', 0.0)
    if not accelerator.is_main_process or rows <= 0:
        return
    jobs = totals.get('train/hindsight_jobs', 0.0)
    applied = totals.get('train/hindsight_applied_positions', 0.0)
    rows_with_apply = totals.get('train/hindsight_rows_with_apply', 0.0)
    correct_rows = totals.get('train/hindsight_correct_rows', 0.0)
    eligible = totals.get('train/hindsight_eligible_left_positions', 0.0)
    candidates = totals.get('train/hindsight_candidate_tokens', 0.0)
    finite_candidates = totals.get('train/hindsight_finite_candidate_tokens', 0.0)
    skipped_correctness = totals.get('train/hindsight_skipped_correctness_only_rows', 0.0)
    apply_rate = applied / jobs if jobs > 0 else 0.0
    rows_apply_rate = rows_with_apply / rows if rows > 0 else 0.0
    logger.info(f'Hindsight correction summary: rows={rows:.0f}, rows_with_apply={rows_with_apply:.0f} ({rows_apply_rate:.3f}), correct_rows={correct_rows:.0f}, skipped_correctness_only={skipped_correctness:.0f}, eligible_left={eligible:.0f}, jobs={jobs:.0f}, applied={applied:.0f} ({apply_rate:.3f}), candidates={candidates:.0f}, finite_candidates={finite_candidates:.0f}')

def _force_zero3_gather_on_model_save(model):
    try:
        model_config = getattr(model, '_config', None)
        zero_config = getattr(model_config, 'zero_config', None)
        if zero_config is not None and hasattr(zero_config, 'gather_16bit_weights_on_model_save'):
            zero_config.gather_16bit_weights_on_model_save = True
    except Exception:
        pass

def _zero_stage_int(stage):
    value = getattr(stage, 'value', stage)
    try:
        return int(value)
    except Exception:
        text = str(value).lower()
        if 'weights' in text or text.endswith('.3') or text == '3':
            return 3
        if 'optimizer' in text or text.endswith('.2') or text == '2':
            return 2
        if 'disabled' in text or text.endswith('.0') or text == '0':
            return 0
    return None

def _deepspeed_zero_stage(model):
    if hasattr(model, 'zero_optimization_stage'):
        try:
            return _zero_stage_int(model.zero_optimization_stage())
        except Exception:
            pass
    model_config = getattr(model, '_config', None)
    if model_config is not None:
        for attr in ('zero_optimization_stage', 'zero_stage'):
            if hasattr(model_config, attr):
                stage = _zero_stage_int(getattr(model_config, attr))
                if stage is not None:
                    return stage
        zero_config = getattr(model_config, 'zero_config', None)
        if zero_config is not None:
            for attr in ('stage', 'zero_stage'):
                if hasattr(zero_config, attr):
                    stage = _zero_stage_int(getattr(zero_config, attr))
                    if stage is not None:
                        return stage
    return None

def _get_model_state_dict_for_save(model, model_to_save, accelerator):
    zero_stage = _deepspeed_zero_stage(model)
    if zero_stage == 3:
        _force_zero3_gather_on_model_save(model)
        if hasattr(model, 'zero_gather_16bit_weights_on_model_save') and (not model.zero_gather_16bit_weights_on_model_save()):
            raise ValueError('DeepSpeed ZeRO-3 model save requires gather_16bit_weights_on_model_save=True')
        return model._zero3_consolidated_16bit_state_dict()
    if zero_stage is not None:
        try:
            from deepspeed.checkpoint.utils import clone_tensors_for_torch_save
            return clone_tensors_for_torch_save(model_to_save.state_dict())
        except Exception:
            return model_to_save.state_dict()
    return accelerator.get_state_dict(model)

def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {'1', 'true', 'yes', 'y', 'on'}:
            return True
        if lowered in {'0', 'false', 'no', 'n', 'off'}:
            return False
    return bool(value)

def _training_value(config, key, default):
    value = OmegaConf.select(config, f'training.{key}', default=None)
    return default if value is None else value

def _copy_pad_embedding_to_mask_on_init(config) -> bool:
    if not _as_bool(OmegaConf.select(config, 'model.copy_pad_embedding_to_mask', default=False)):
        return False
    if not _as_bool(OmegaConf.select(config, 'experiment.start_from_scratch', default=True)):
        return False
    current_epoch = 1
    return current_epoch <= 1

def _maybe_copy_pad_embedding_to_mask(model, tokenizer, config):
    if not _copy_pad_embedding_to_mask_on_init(config):
        return
    copy_pad_embedding_to_mask(model, tokenizer=tokenizer, log_fn=logger.info)

def _teacher_block_size_is_unset(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {'', 'null', 'none'}:
            return True
        return stripped == '-1'
    return int(value) == -1

def _teacher_block_size(config) -> int:
    student_block_size = int(_training_value(config, 'block_size', 0))
    teacher_block_size = 1
    teacher_block_size = student_block_size if _teacher_block_size_is_unset(teacher_block_size) else int(teacher_block_size)
    if teacher_block_size <= 0:
        raise ValueError('training.teacher_block_size must be positive or -1')
    return teacher_block_size

def _hindsight_scorer_path(config):
    path = OmegaConf.select(config, 'model.hindsight_scorer_model', default=None)
    if path in (None, '', 'null', 'None'):
        path = OmegaConf.select(config, 'model.teacher_ar_model', default=None)
    if path in (None, '', 'null', 'None'):
        path = OmegaConf.select(config, 'model.teacher_model', default=None)
    return path

def _compute_token_divergence(teacher_logprobs, student_logprobs, config):
    k = int(config.training.top_k_logits)
    if k > 0:
        teacher_logprobs, indices = teacher_logprobs.topk(min(k, teacher_logprobs.size(-1)), dim=-1)
        student_logprobs = student_logprobs.gather(-1, indices)
    finite = torch.isfinite(teacher_logprobs)
    terms = torch.where(finite, teacher_logprobs.exp() * (teacher_logprobs - student_logprobs), torch.zeros_like(teacher_logprobs))
    return terms.sum(dim=-1)

def _load_local_hindsight_scorer(scorer_path, device):
    from transformers.integrations import deepspeed as hf_deepspeed
    saved_hf_ds_ref = getattr(hf_deepspeed, '_hf_deepspeed_config_weak_ref', None)
    hf_deepspeed.unset_hf_deepspeed_config()
    try:
        scorer_model = AutoModelForCausalLM.from_pretrained(scorer_path, trust_remote_code=True, torch_dtype='auto')
    finally:
        hf_deepspeed._hf_deepspeed_config_weak_ref = saved_hf_ds_ref
    if hasattr(scorer_model, 'config'):
        scorer_model.config.fuse_cross_entropy = False
        scorer_model.config.use_cache = True
    scorer_model.requires_grad_(False)
    scorer_model = scorer_model.to(device)
    scorer_model.eval()
    return scorer_model

@torch.no_grad()
def _build_aligned_ar_cache(ar_model, input_ids, position_ids, pad_id):
    from transformers.cache_utils import DynamicCache
    if input_ids.ndim != 1 or position_ids.ndim != 1:
        raise ValueError('aligned AR cache expects 1-D input_ids and position_ids')
    if input_ids.shape[0] != position_ids.shape[0]:
        raise ValueError('input_ids and position_ids must have the same length')
    if input_ids.shape[0] == 0:
        return DynamicCache()
    outputs = ar_model.model(input_ids=input_ids.unsqueeze(0), attention_mask=input_ids.ne(pad_id).unsqueeze(0), position_ids=position_ids.unsqueeze(0), use_cache=True)
    cache = outputs.past_key_values
    if not isinstance(cache, DynamicCache):
        raise TypeError(f'Expected DynamicCache from teacher, got {type(cache).__name__}')
    return cache

@torch.no_grad()
def _score_aligned_future_branches(ar_model, shared_cache, shared_sequence, shared_position_ids, prefix_length, branch_pos, branch_token_ids, future_start, future_end, pad_id, max_chunk_size, max_cache_bytes=16 * 1024 ** 3):
    from transformers.cache_utils import DynamicCache
    if shared_sequence.ndim != 1 or shared_position_ids.ndim != 1:
        raise ValueError('shared_sequence and shared_position_ids must be 1-D')
    branch_token_ids = torch.as_tensor(branch_token_ids, dtype=torch.long, device=shared_sequence.device)
    if branch_token_ids.ndim != 1:
        raise ValueError('branch_token_ids must be 1-D')
    if branch_token_ids.numel() == 0:
        return torch.empty(0, device=shared_sequence.device, dtype=torch.float32)
    prefix_length = int(prefix_length)
    branch_pos = int(branch_pos)
    future_start = int(future_start)
    future_end = int(future_end)
    score_start = max(future_start - 1, 0)
    score_end = max(future_end - 1, 0)
    if score_end <= score_start:
        return torch.zeros(branch_token_ids.shape[0], device=shared_sequence.device)
    if not 0 <= prefix_length <= branch_pos <= score_start:
        raise ValueError(f'invalid branch scoring span: prefix={prefix_length}, branch={branch_pos}, score=[{score_start}, {score_end})')
    if score_end >= shared_sequence.shape[0]:
        raise ValueError('future span target is outside shared_sequence')
    if prefix_length > shared_cache.get_seq_length():
        raise ValueError('prefix_length is longer than the shared cache')
    if max_chunk_size <= 0 or max_cache_bytes <= 0:
        raise ValueError('cache batching limits must be positive')
    if prefix_length == 0:
        cache_limited_chunk_size = max_chunk_size
    else:
        bytes_per_prefix_token = sum(((key[0, :, :1, :].numel() + value[0, :, :1, :].numel()) * key.element_size() for key, value in shared_cache))
        cache_limited_chunk_size = max(1, max_cache_bytes // (bytes_per_prefix_token * prefix_length))
    chunk_size = min(max_chunk_size, cache_limited_chunk_size)
    scores = []
    suffix_base = shared_sequence[prefix_length:score_end]
    visible_base = shared_sequence[:score_end]
    suffix_pos = shared_position_ids[prefix_length:score_end]
    relative_logit_positions = torch.arange(score_start - prefix_length, score_end - prefix_length, device=shared_sequence.device)
    cache_position = torch.arange(prefix_length, score_end, device=shared_sequence.device)
    targets = shared_sequence[score_start + 1:score_end + 1]
    for chunk_start in range(0, branch_token_ids.numel(), chunk_size):
        chunk = branch_token_ids[chunk_start:chunk_start + chunk_size]
        batch_size = chunk.shape[0]
        if prefix_length == 0:
            branch_cache = DynamicCache()
        else:
            repeated = tuple(((key[:, :, :prefix_length, :].expand(batch_size, -1, -1, -1), value[:, :, :prefix_length, :].expand(batch_size, -1, -1, -1)) for key, value in shared_cache))
            branch_cache = DynamicCache.from_legacy_cache(repeated)
        suffix = suffix_base.unsqueeze(0).repeat(batch_size, 1)
        suffix[:, branch_pos - prefix_length] = chunk
        visible_tokens = visible_base.unsqueeze(0).repeat(batch_size, 1)
        visible_tokens[:, branch_pos] = chunk
        attention_mask = visible_tokens.ne(pad_id)
        position_ids = suffix_pos.unsqueeze(0).repeat(batch_size, 1)
        logits = ar_model(input_ids=suffix, attention_mask=attention_mask, position_ids=position_ids, cache_position=cache_position, past_key_values=branch_cache, use_cache=True, logits_to_keep=relative_logit_positions).logits
        log_probs = F.log_softmax(logits.float(), dim=-1)
        chunk_targets = targets.unsqueeze(0).expand(batch_size, -1)
        target_mask = chunk_targets.ne(pad_id)
        token_logp = log_probs.gather(-1, chunk_targets.unsqueeze(-1)).squeeze(-1)
        scores.append(torch.where(target_mask, token_logp, torch.zeros_like(token_logp)).sum(dim=-1))
    return torch.cat(scores, dim=0)

@torch.no_grad()
def _apply_aligned_hindsight_correction(teacher_logprobs, student_logprobs, teacher_model, hindsight_scorer_model, extended_input_ids, tok_idx_ext, L0, L1, tokenizer, config, rewards=None, hindsight_correctness=None, stats=None):
    if stats is not None:
        _reset_hindsight_stats(stats, rows=teacher_logprobs.shape[0])
    hindsight_lambda = float(_training_value(config, 'hindsight_lambda', 0.0))
    if hindsight_lambda == 0.0:
        return teacher_logprobs
    if teacher_model is None:
        raise ValueError('training.hindsight_lambda requires model.teacher_model')
    student_block_size = int(_training_value(config, 'block_size', 0))
    teacher_block_size = _teacher_block_size(config)
    if student_block_size <= teacher_block_size:
        raise ValueError(f'training.hindsight_lambda expects student block size to be larger than teacher block size: teacher_block_size={teacher_block_size}, block_size={student_block_size}')
    if hindsight_scorer_model is None:
        raise ValueError('training.hindsight_lambda requires model.hindsight_scorer_model')
    top_k = int(_training_value(config, 'hindsight_top_k', 16))
    if top_k <= 0:
        raise ValueError('training.hindsight_top_k must be positive when hindsight_lambda != 0')
    require_visible_future = _as_bool(_training_value(config, 'hindsight_require_visible_future', _training_value(config, 'hindsight_visible_only', True)))
    preserve_zero_support = _as_bool(_training_value(config, 'hindsight_preserve_zero_support', True))
    correctness_only = _as_bool(_training_value(config, 'hindsight_correctness_only', _training_value(config, 'hindsight_correct_only', False)))
    max_chunk_size = int(_training_value(config, 'hindsight_score_chunk_size', 128))
    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError('training.hindsight_lambda requires tokenizer.mask_token_id')
    prompt = extended_input_ids[:, :L0]
    prompt_response = extended_input_ids[:, :L0 + L1]
    response_noised = extended_input_ids[:, L0 + L1:]
    prompt_mask = torch.cat([torch.zeros_like(prompt), torch.ones_like(response_noised)], dim=1).bool()
    prompt_noised_response = torch.cat([prompt, response_noised], dim=1)
    pad_mask = prompt_noised_response.ne(pad_id)
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    if im_end_id is None:
        im_end_mask_before_exclude = torch.ones_like(prompt_mask, dtype=torch.bool)
        is_response_im_end = torch.zeros_like(prompt_mask, dtype=torch.bool)
    else:
        is_im_end = prompt_response.eq(im_end_id)
        is_response_im_end = is_im_end & prompt_mask
        im_end_cumsum = is_response_im_end.cumsum(dim=1)
        im_end_shifted = F.pad(im_end_cumsum[:, :-1], (1, 0))
        im_end_mask_before_exclude = im_end_shifted.eq(0)
    im_end_mask = im_end_mask_before_exclude
    if getattr(config.training, 'exclude_im_end', False):
        im_end_mask = im_end_mask & ~is_response_im_end
    response_mask = prompt_mask & pad_mask & im_end_mask
    future_context_mask = prompt_mask & pad_mask & im_end_mask_before_exclude
    masked_token_mask = prompt_noised_response.eq(mask_id)
    loss_mask = response_mask & masked_token_mask
    ar_teacher = getattr(hindsight_scorer_model, 'module', hindsight_scorer_model)
    corrected = teacher_logprobs.clone()
    student_logprobs_detached = student_logprobs.detach()
    vocab_size = teacher_logprobs.shape[-1]
    scorer_vocab_size = int(getattr(getattr(ar_teacher, 'config', None), 'vocab_size', vocab_size))
    B = extended_input_ids.shape[0]
    correctness_filter = None
    if hindsight_correctness is not None:
        correctness_filter = torch.as_tensor(hindsight_correctness, device=teacher_logprobs.device, dtype=torch.bool).detach().reshape(-1)
        if correctness_filter.numel() != B:
            raise ValueError(f'hindsight correctness length mismatch: got {correctness_filter.numel()}, expected {B}')
    elif rewards is not None:
        correctness_filter = torch.as_tensor(rewards, device=teacher_logprobs.device, dtype=torch.float32).detach().reshape(-1) > 0
        if correctness_filter.numel() != B:
            raise ValueError(f'hindsight rewards length mismatch: got {correctness_filter.numel()}, expected {B}')
    if correctness_filter is not None:
        if stats is not None:
            stats['correct_rows'] = int(correctness_filter.sum().item())
    elif correctness_only:
        raise ValueError('training.hindsight_correctness_only requires per-row correctness or rewards')
    for batch_idx in range(B):
        if correctness_only and (not bool(correctness_filter[batch_idx].item())):
            if stats is not None:
                stats['skipped_correctness_only_rows'] += 1
            continue
        sequence = prompt_response[batch_idx].detach()
        position_ids = tok_idx_ext[batch_idx, :L0 + L1].detach()
        jobs = []
        row_applied = 0
        for block_start in range(0, L1, student_block_size):
            block_end = min(block_start + student_block_size, L1)
            for chunk_start in range(block_start, block_end, teacher_block_size):
                chunk_end = min(chunk_start + teacher_block_size, block_end)
                left_positions = range(chunk_start, chunk_end)
                right_positions = list(range(chunk_end, block_end))
                if not right_positions:
                    continue
                future_start_abs = L0 + right_positions[0]
                future_end_abs = L0 + right_positions[-1] + 1
                visible_right = [right_rel for right_rel in right_positions if future_context_mask[batch_idx, L0 + right_rel].item() and response_noised[batch_idx, right_rel].item() != mask_id]
                if require_visible_future and (not visible_right):
                    if stats is not None:
                        stats['skipped_no_visible_blocks'] += 1
                    continue
                if not future_context_mask[batch_idx, future_start_abs:future_end_abs].any().item():
                    if stats is not None:
                        stats['skipped_no_future_blocks'] += 1
                    continue
                for left_rel in left_positions:
                    left_abs = L0 + left_rel
                    if not loss_mask[batch_idx, left_abs].item():
                        if stats is not None:
                            stats['skipped_no_loss_positions'] += 1
                        continue
                    if stats is not None:
                        stats['eligible_left_positions'] += 1
                    k = min(top_k, vocab_size)
                    teacher_idx = torch.topk(teacher_logprobs[batch_idx, left_abs].detach(), k, dim=-1).indices
                    student_idx = torch.topk(student_logprobs_detached[batch_idx, left_abs], k, dim=-1).indices
                    candidates = torch.unique(torch.cat([teacher_idx, student_idx], dim=0))
                    candidates = candidates[candidates < scorer_vocab_size]
                    if candidates.numel() == 0:
                        if stats is not None:
                            stats['skipped_no_candidates'] += 1
                        continue
                    if stats is not None:
                        stats['jobs'] += 1
                        stats['candidate_tokens'] += int(candidates.numel())
                        stats['scored_branches'] += int(candidates.numel()) + 1
                    clean_left = prompt_response[batch_idx, left_abs]
                    branch_tokens = torch.cat([clean_left.reshape(1), candidates])
                    jobs.append((left_abs, candidates, branch_tokens, L0 + chunk_start, future_start_abs, future_end_abs))
        if not jobs:
            continue
        if stats is not None:
            stats['rows_with_jobs'] += 1
        shared_cache = _build_aligned_ar_cache(ar_teacher, sequence, position_ids, pad_id)
        for left_abs, candidates, branch_tokens, prefix_length, future_start_abs, future_end_abs in jobs:
            scores = _score_aligned_future_branches(ar_teacher, shared_cache, sequence, position_ids, prefix_length=prefix_length, branch_pos=left_abs, branch_token_ids=branch_tokens.to(sequence.device), future_start=future_start_abs, future_end=future_end_abs, pad_id=pad_id, max_chunk_size=max_chunk_size, max_cache_bytes=int(_training_value(config, 'hindsight_cache_max_bytes', 16 * 1024 ** 3)))
            baseline_score = scores[0]
            candidate_scores = scores[1:]
            deltas = candidate_scores - baseline_score
            finite = torch.isfinite(deltas)
            if not finite.any().item():
                if stats is not None:
                    stats['skipped_nonfinite_jobs'] += 1
                continue
            candidates_finite = candidates[finite]
            deltas_finite = deltas[finite]
            if stats is not None:
                stats['finite_candidate_tokens'] += int(candidates_finite.numel())
            pos_probs = corrected[batch_idx, left_abs].exp()
            pos_probs[candidates_finite] = pos_probs[candidates_finite] * torch.exp(hindsight_lambda * deltas_finite).to(pos_probs.dtype)
            normalizer = pos_probs.sum()
            if torch.isfinite(normalizer).item() and normalizer.item() > 0:
                normalized = pos_probs / normalizer
                if preserve_zero_support:
                    corrected[batch_idx, left_abs] = normalized.log()
                else:
                    corrected[batch_idx, left_abs] = normalized.clamp_min(1e-12).log()
                row_applied += 1
                if stats is not None:
                    stats['applied_positions'] += 1
            elif stats is not None:
                stats['skipped_bad_normalizer_jobs'] += 1
        if row_applied > 0 and stats is not None:
            stats['rows_with_apply'] += 1
    return corrected

def _combine_rounds_one_state_per_block(per_prompt_ext, per_prompt_pm, clean_input_ids, L0, L1, block_size):
    """Stitch a trajectory's T denoising rounds into a single training row,
    independently sampling one round per block for the block's tail + pmask.
    The prompt and clean response are invariant across rounds.
    """
    ext = torch.stack(per_prompt_ext, dim=0)
    pm = torch.stack(per_prompt_pm, dim=0)
    tail = clean_input_ids[L0:L0 + L1].clone()
    out_pm = torch.zeros(L0 + L1, dtype=torch.bool, device=pm.device)
    for bs in range(0, L1, block_size):
        be = min(bs + block_size, L1)
        rounds = pm[:, L0 + bs:L0 + be].any(dim=1).nonzero().flatten().tolist()
        if not rounds:
            continue
        r = random.choice(rounds)
        out_pm[L0 + bs:L0 + be] = pm[r, L0 + bs:L0 + be]
        tail[bs:be] = ext[r, L0 + L1 + bs:L0 + L1 + be]
    return (torch.cat([clean_input_ids, tail], dim=0), out_pm)

class TrainDataset(Dataset):

    def __init__(self, extended_input_ids, p_mask, tok_idx_ext, labels, reward, hindsight_correctness=None):
        self.extended_input_ids = extended_input_ids
        self.p_mask = p_mask
        self.tok_idx_ext = tok_idx_ext
        self.labels = labels
        self.reward = reward
        if hindsight_correctness is None:
            hindsight_correctness = [float(r) > 0 for r in reward]
        self.hindsight_correctness = hindsight_correctness
        self.logp_old_tok = torch.full((len(extended_input_ids), p_mask.shape[1]), -1e309)

    def __len__(self):
        return len(self.extended_input_ids)

    def __getitem__(self, idx):
        return (idx, self.extended_input_ids[idx], self.p_mask[idx], self.tok_idx_ext[idx], self.labels[idx], self.reward[idx], self.hindsight_correctness[idx])

def _checkpoint_save_policy(config):
    """Return (save named epoch checkpoint, persist full state) for this step."""
    current_step = int(config.experiment.current_epoch)
    save_every = int(config.experiment.get('save_every', 10))
    periodic = save_every > 0 and current_step % save_every == 0
    pinned_steps = {int(step) for step in config.experiment.get('checkpoint_steps_to_keep', [])}
    is_pinned = current_step in pinned_steps
    is_final = False
    if bool(config.experiment.get('save_final_epoch_checkpoint', True)):
        for key in ('stop_RL_step', 'total_step'):
            endpoint = config.experiment.get(key, -1)
            if endpoint not in (-1, None) and current_step == int(endpoint):
                is_final = True
                break
    return (periodic or is_pinned or is_final, periodic or is_final)

def save_checkpoint(model, tokenizer, config, accelerator, name, save_training_state_flag=True, lr_scheduler=None):
    from pathlib import Path
    import time, json, shutil, os, glob, importlib, inspect
    output_dir = Path(config.experiment.project)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_total_limit = config.experiment.get('checkpoints_total_limit', None)
    if accelerator.is_main_process and checkpoints_total_limit is not None:
        ckpts = sorted([d for d in output_dir.iterdir() if d.name.startswith('checkpoint')], key=lambda p: int(p.name.split('-')[1]))
        if len(ckpts) >= checkpoints_total_limit:
            to_remove = ckpts[:len(ckpts) - checkpoints_total_limit + 1]
            logger.info(f"removing checkpoints: {', '.join((p.name for p in to_remove))}")
            for p in to_remove:
                shutil.rmtree(p, ignore_errors=True)
    save_base = output_dir / 'ckpt'
    save_base.mkdir(exist_ok=True)
    accelerator.wait_for_everyone()
    if save_training_state_flag:
        save_training_state(model, config, name, lr_scheduler=lr_scheduler)
    accelerator.wait_for_everyone()
    model_to_save = accelerator.unwrap_model(model)
    state_dict = _get_model_state_dict_for_save(model, model_to_save, accelerator)
    if accelerator.is_main_process:
        save_dir = save_base / name
        tmp_dir = save_base / f'{name}.tmp'
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        model_to_save.save_pretrained(tmp_dir, save_function=accelerator.save, state_dict=state_dict, safe_serialization=True)
        ensure_diffusion_mask_token(tokenizer)
        tokenizer.save_pretrained(str(tmp_dir))

        def _copy_dynamic_modules(dst_dir, model_obj, tok_obj):
            copied = 0
            modules = set()
            for obj in [model_obj, getattr(model_obj, 'config', None), tok_obj]:
                if obj is None:
                    continue
                modname = getattr(obj.__class__, '__module__', None)
                if modname:
                    modules.add(modname)
            for modname in modules:
                try:
                    mod = importlib.import_module(modname)
                    src_file = inspect.getsourcefile(mod)
                    if not src_file or not os.path.exists(src_file):
                        continue
                    base_dir = os.path.dirname(src_file)
                    for pattern in ('modeling_*.py', 'configuration_*.py', 'tokenization_*.py', 'processing_*.py'):
                        for fn in glob.glob(os.path.join(base_dir, pattern)):
                            dst = os.path.join(dst_dir, os.path.basename(fn))
                            if os.path.exists(dst):
                                continue
                            shutil.copy2(fn, dst)
                            copied += 1
                except Exception as e:
                    logger.warning(f'Skip copying from module {modname}: {e}')
            logger.info(f'Copied {copied} custom module files into {dst_dir}')
        _copy_dynamic_modules(str(tmp_dir), model_to_save, tokenizer)
        if save_dir.exists():
            old_dir = save_base / f'{name}.old'
            if old_dir.exists():
                shutil.rmtree(old_dir, ignore_errors=True)
            os.rename(save_dir, old_dir)
            os.rename(tmp_dir, save_dir)
            shutil.rmtree(old_dir, ignore_errors=True)
        else:
            os.rename(tmp_dir, save_dir)
        metadata = {'save_time': time.strftime('%Y-%m-%d %H:%M:%S'), 'current_epoch': config.experiment.current_epoch, 'last_save_name': name}
        with (save_base / 'metadata.json').open('w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f'Saved model + tokenizer to {save_dir}')

def prune_epoch_checkpoints(config, accelerator):
    keep = int(config.experiment.get('epoch_checkpoints_to_keep', 1))
    pinned_steps = {int(step) for step in config.experiment.get('checkpoint_steps_to_keep', [])}
    if keep < 0:
        return
    if not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        return
    save_base = Path(config.experiment.project) / 'ckpt'
    if not save_base.is_dir():
        accelerator.wait_for_everyone()
        return
    epoch_ckpts = []
    for path in save_base.iterdir():
        if not path.is_dir():
            continue
        name = path.name
        if not name.startswith('epoch-') or name.endswith(('.tmp', '.old')):
            continue
        try:
            step = int(name.split('-', 1)[1])
        except ValueError:
            continue
        epoch_ckpts.append((step, path))
    epoch_ckpts.sort(key=lambda item: item[0])
    unpinned_ckpts = [item for item in epoch_ckpts if item[0] not in pinned_steps]
    to_remove = unpinned_ckpts[:max(0, len(unpinned_ckpts) - keep)]
    for _, path in to_remove:
        logger.info(f'Removing old epoch checkpoint {path}')
        shutil.rmtree(path, ignore_errors=True)
    accelerator.wait_for_everyone()

def get_training_state_dir(config, name):
    return Path(config.experiment.project) / 'training_state' / name

def load_training_state(model, config, name):
    training_state_dir = get_training_state_dir(config, name)
    if not training_state_dir.is_dir():
        logger.warning(f'No persisted training state found at {training_state_dir}. Starting with a fresh optimizer.')
        return False
    if not hasattr(model, 'load_checkpoint'):
        logger.warning('Prepared model does not expose DeepSpeed checkpoint loading. Skipping optimizer restore.')
        return False
    load_path, client_state = model.load_checkpoint(str(training_state_dir), tag=TRAINING_STATE_TAG)
    if load_path is None:
        logger.warning(f'DeepSpeed did not restore training state from {training_state_dir}. Starting with a fresh optimizer.')
        return False
    if config.experiment.get('require_training_state', False):
        expected_step = int(config.experiment.current_epoch) - 1
        saved_step = (client_state or {}).get('current_epoch')
        if saved_step is None or int(saved_step) != expected_step:
            raise RuntimeError(f'Resume state is at step {saved_step}, but next step {config.experiment.current_epoch} requires state from {expected_step}.')
    logger.info(f'Restored DeepSpeed training state from {load_path}')
    return True

def save_training_state(model, config, name, lr_scheduler=None):
    if not hasattr(model, 'save_checkpoint'):
        logger.warning('Prepared model does not expose DeepSpeed checkpoint saving. Skipping optimizer persistence.')
        return
    training_state_dir = get_training_state_dir(config, name)
    parent_dir = training_state_dir.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = parent_dir / f'{training_state_dir.name}.tmp'
    old_dir = parent_dir / f'{training_state_dir.name}.old'
    try:
        import torch.distributed as _dist
        is_main = not _dist.is_initialized() or _dist.get_rank() == 0
        _barrier = _dist.barrier if _dist.is_initialized() else lambda: None
    except Exception:
        is_main = True
        _barrier = lambda: None
    if is_main:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
    _barrier()
    model.save_checkpoint(str(tmp_dir), tag=TRAINING_STATE_TAG, client_state={'save_time': time.strftime('%Y-%m-%d %H:%M:%S'), 'current_epoch': config.experiment.current_epoch})
    if is_main and lr_scheduler is not None:
        lr_path = tmp_dir / 'lr_scheduler.pt'
        torch.save(lr_scheduler.state_dict(), lr_path)
        logger.info(f'Saved LR scheduler state to {lr_path}')
    _barrier()
    if is_main:
        if old_dir.exists():
            shutil.rmtree(old_dir, ignore_errors=True)
        if training_state_dir.exists():
            os.rename(training_state_dir, old_dir)
        os.rename(tmp_dir, training_state_dir)
        shutil.rmtree(old_dir, ignore_errors=True)
        logger.info(f'Saved DeepSpeed training state to {training_state_dir}')
    _barrier()

def load_lr_scheduler_state(lr_scheduler, config, name):
    training_state_dir = get_training_state_dir(config, name)
    lr_path = training_state_dir / 'lr_scheduler.pt'
    if not lr_path.is_file():
        logger.warning(f'No LR scheduler state at {lr_path}. Starting scheduler from step 0.')
        return False
    state_dict = torch.load(str(lr_path), map_location='cpu')
    lr_scheduler.load_state_dict(state_dict)
    logger.info(f'Restored LR scheduler state from {lr_path}')
    return True

def init_training(config):
    """One-time initialization of the training engine.

    Creates Accelerator, loads student + teacher models, optimizer, LR scheduler.
    Returns a state dict used by train_one_step().
    Must be called inside an `accelerate launch` distributed context.
    """
    wandb_enabled = bool(config.wandb.get('enabled', True))
    project_name = config.experiment.project
    if config.experiment.current_epoch > 1 or not config.experiment.start_from_scratch:
        _resume_model_name = config.experiment.get('resume_model_name', None) or config.model.optimized_name
        pretrained_model = os.path.join(project_name, 'ckpt', _resume_model_name)
        if not os.path.exists(pretrained_model):
            logger.warning(f'Resume requested but checkpoint not found at {pretrained_model}; falling back to {config.model.pretrained_model}')
            pretrained_model = config.model.pretrained_model
    else:
        pretrained_model = config.model.pretrained_model
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    config.experiment.logging_dir = str(Path(config.experiment.project) / 'logs')
    _zero_stage = int(config.training.get('zero_stage', 3))
    _offload_opt = config.training.get('offload_optimizer_device', 'cpu')
    _offload_param = config.training.get('offload_param_device', 'cpu')
    _student_kwargs = dict(gradient_accumulation_steps=config.training.gradient_accumulation_steps, gradient_clipping=config.training.max_grad_norm, zero_stage=_zero_stage, offload_optimizer_device=_offload_opt)
    if _zero_stage == 3:
        _student_kwargs.update(offload_param_device=_offload_param, zero3_init_flag=True, zero3_save_16bit_model=True)
    _teacher_kwargs = dict(zero_stage=3, offload_param_device=_offload_param, zero3_init_flag=True, zero3_save_16bit_model=True)
    deepspeed_plugins = {'student': DeepSpeedPlugin(**_student_kwargs), 'teacher': DeepSpeedPlugin(**_teacher_kwargs)}
    accelerator = Accelerator(gradient_accumulation_steps=config.training.gradient_accumulation_steps, mixed_precision=config.training.mixed_precision, log_with='wandb' if wandb_enabled else None, project_dir=config.experiment.logging_dir, split_batches=False, deepspeed_plugins=deepspeed_plugins)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()
    if accelerator.is_main_process and wandb_enabled:
        run_id = config.wandb.get('run_id', None) or os.getenv('WANDB_RUN_ID', None)
        if run_id is None:
            raise ValueError('WANDB_RUN_ID environment variable is not set.')
        wandb_init_kwargs = dict(id=run_id, resume=str(config.wandb.get('resume', 'allow')))
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop('experiment.resume_from_checkpoint', None)
        wandb_project = config.wandb.get('project') or config.experiment.project
        accelerator.init_trackers(wandb_project, config=wandb_config, init_kwargs={'wandb': wandb_init_kwargs})
        wandb.define_metric('train/current_epoch')
        wandb.define_metric('*', step_metric='train/current_epoch')
    if accelerator.is_main_process:
        os.makedirs(config.experiment.project, exist_ok=True)
        config_path = Path(config.experiment.project) / 'config.yaml'
        logging.info(f'Saving config to {config_path}')
        OmegaConf.save(config, config_path)
    _env_seed = os.environ.get('DOPD_SEED')
    _step_seed = int(_env_seed) if _env_seed is not None else config.training.seed
    if _step_seed is not None:
        set_seed(_step_seed)
    logger.info('Loading models and optimizer')
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
    ensure_diffusion_mask_token(tokenizer)
    uni_prompting = UniversalPrompting(tokenizer, max_prompt_len=config.training.max_prompt_len, max_gen_length=config.training.max_gen_length, ignore_id=-100)
    from dopd.modeling import _register_a2d_model_classes
    from transformers import AutoModelForMaskedLM
    _register_a2d_model_classes()
    model = AutoModelForMaskedLM.from_pretrained(pretrained_model, trust_remote_code=False, torch_dtype='auto')
    _maybe_copy_pad_embedding_to_mask(model, tokenizer, config)
    if hasattr(model, 'config'):
        model.config.fuse_cross_entropy = False
    if config.training.gradient_checkpointing_enable:
        model.gradient_checkpointing_enable()
        if hasattr(model, 'config'):
            model.config.use_cache = False
    else:
        model = model.to(accelerator.device)
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        mask_id = tokenizer.pad_token_id
    pad_id = tokenizer.pad_token_id
    optimizer_config = config.optimizer.params
    no_decay = ['bias', 'layer_norm.weight', 'mlm_ln.weight', 'embeddings.weight']
    optimizer_grouped_parameters = [{'params': [p for n, p in model.named_parameters() if p.requires_grad and (not any((nd in n for nd in no_decay)))], 'weight_decay': optimizer_config.weight_decay}, {'params': [p for n, p in model.named_parameters() if p.requires_grad and any((nd in n for nd in no_decay))], 'weight_decay': 0.0}]
    optimizer_type = config.optimizer.name
    if optimizer_type == 'adamw':
        optimizer = AdamW(optimizer_grouped_parameters, lr=optimizer_config.learning_rate, betas=(optimizer_config.beta1, optimizer_config.beta2), weight_decay=optimizer_config.weight_decay, eps=optimizer_config.epsilon)
    else:
        raise ValueError(f'Optimizer {optimizer_type} not supported')
    _student_micro_batch = int(config.training.batch_size_lm)
    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = _student_micro_batch
    model, optimizer = accelerator.prepare(model, optimizer)
    should_resume = config.experiment.current_epoch > 1 or not config.experiment.start_from_scratch
    if should_resume:
        restored = load_training_state(model, config, config.model.optimized_name)
        if config.experiment.get('require_training_state', False) and (not restored):
            raise RuntimeError('Required optimizer state could not be restored; refusing to restart it.')
    accelerator.state.select_deepspeed_plugin('teacher')
    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1
    teacher_model = AutoModelForCausalLM.from_pretrained(config.model.teacher_model, trust_remote_code=True, torch_dtype='auto')
    if hasattr(teacher_model, 'config'):
        teacher_model.config.fuse_cross_entropy = False
    teacher_model.requires_grad_(False)
    teacher_model = accelerator.prepare(teacher_model)
    teacher_model.eval()
    accelerator.state.select_deepspeed_plugin('student')
    hindsight_scorer_model = None
    if 'kl' in ('kl', 'jsd') and abs(float(_training_value(config, 'hindsight_lambda', 0.0))) > 0.0:
        scorer_path = _hindsight_scorer_path(config)
        if scorer_path in (None, '', 'null', 'None'):
            raise ValueError('training.hindsight_lambda requires model.hindsight_scorer_model')
        hindsight_scorer_model = _load_local_hindsight_scorer(scorer_path, accelerator.device)
        logger.info(f'Loaded local hindsight AR scorer from {scorer_path}')
    _student_total_params = sum((getattr(p, 'ds_numel', p.numel()) for p in model.parameters()))
    _teacher_total_params = sum((getattr(p, 'ds_numel', p.numel()) for p in teacher_model.parameters())) if teacher_model is not None else 0
    _scorer_total_params = sum((p.numel() for p in hindsight_scorer_model.parameters())) if hindsight_scorer_model is not None else 0
    logger.info(f'Param counts: student={_student_total_params / 1000000.0:.1f}M, teacher={_teacher_total_params / 1000000.0:.1f}M, scorer={_scorer_total_params / 1000000.0:.1f}M')
    _token_count_file = os.path.join(config.experiment.project, 'temp_data', 'cumulative_loss_tokens.txt')
    if os.path.exists(_token_count_file):
        with open(_token_count_file) as f:
            cumulative_loss_tokens = int(f.read().strip())
    else:
        cumulative_loss_tokens = 0
    _block_size = config.training.block_size
    _num_tasks = config.rollout.num_task_per_step
    _rounds_per_prompt = 1
    _est_rows = _num_tasks * _rounds_per_prompt
    _total_batch_size = config.training.batch_size_lm * config.training.gradient_accumulation_steps
    _total_batch_size *= accelerator.num_processes
    _num_inner_epochs = config.training.num_train_epochs
    _est_steps_per_rl_step = math.ceil(_est_rows / _total_batch_size) * _num_inner_epochs
    total_rl_steps = config.experiment.total_step
    _scheduler_name = str(config.lr_scheduler.scheduler).lower()
    if total_rl_steps > 0:
        max_train_steps = _est_steps_per_rl_step * total_rl_steps + 1
    elif _scheduler_name in ('constant', 'constant_with_warmup'):
        max_train_steps = None
    else:
        raise ValueError(f"lr_scheduler='{_scheduler_name}' needs a finite training horizon, but experiment.total_step<=0 and no epoch-mode override is active (set dataset.num_data_epochs>=1 or experiment.total_step>0, or use a constant scheduler).")
    _decay_steps = getattr(config.lr_scheduler.params, 'decay_steps', None)
    if _decay_steps is not None:
        _decay_steps = int(_decay_steps) * _est_steps_per_rl_step
    _warmup_rl_steps = config.lr_scheduler.params.warmup_steps
    _warmup_inner_steps = int(_warmup_rl_steps) * _est_steps_per_rl_step
    lr_scheduler = get_scheduler(config.lr_scheduler.scheduler, optimizer=optimizer, num_training_steps=max_train_steps, num_warmup_steps=_warmup_inner_steps, min_lr_scale=config.lr_scheduler.params.min_lr_scale, decay_steps=_decay_steps)
    lr_scheduler = accelerator.prepare(lr_scheduler)
    logger.info(f'LR scheduler: {config.lr_scheduler.scheduler}, warmup={_warmup_inner_steps} inner steps, total={max_train_steps} inner steps, est {_est_steps_per_rl_step} inner steps/RL step')
    if should_resume:
        restored = load_lr_scheduler_state(lr_scheduler, config, config.model.optimized_name)
        if config.experiment.get('require_training_state', False) and (not restored):
            raise RuntimeError('Required LR scheduler state could not be restored; refusing to restart it.')
    return {'accelerator': accelerator, 'model': model, 'teacher_model': teacher_model, 'hindsight_scorer_model': hindsight_scorer_model, 'tokenizer': tokenizer, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'uni_prompting': uni_prompting, 'mask_id': mask_id, 'pad_id': pad_id, 'model_base': 'bd3lm', 'student_total_params': _student_total_params, 'teacher_total_params': _teacher_total_params, 'cumulative_loss_tokens': cumulative_loss_tokens, 'wandb_enabled': wandb_enabled}

def train_one_step(state, config):
    """Run one training step using persistent state from init_training().

    Reads data from {project_name}/temp_data/{optimization_data}.json,
    trains for num_train_epochs, saves checkpoint.
    """
    accelerator = state['accelerator']
    model = state['model']
    teacher_model = state['teacher_model']
    hindsight_scorer_model = state.get('hindsight_scorer_model')
    tokenizer = state['tokenizer']
    optimizer = state['optimizer']
    lr_scheduler = state['lr_scheduler']
    uni_prompting = state['uni_prompting']
    uni_prompting.max_gen_length = config.training.max_gen_length
    mask_id = state['mask_id']
    pad_id = state['pad_id']
    _student_total_params = state['student_total_params']
    _teacher_total_params = state['teacher_total_params']
    cumulative_loss_tokens = state['cumulative_loss_tokens']
    wandb_enabled = state['wandb_enabled']
    project_name = config.experiment.project
    current_epoch = config.experiment.current_epoch
    data_path = project_name + '/temp_data/' + config.dataset.optimization_data + '.json'
    with open(data_path, 'r') as f:
        dataset_load = json.load(f)
    if len(dataset_load) == 0:
        logger.warning('No training data after filtering. Skipping this training step.')
        return
    prompt_list = []
    response_list = []
    step_map_list = []
    reward_list = []
    hindsight_correctness_list = []
    for x in dataset_load:
        prompt_list.append(x['prompt'])
        response_list.append(x['response'])
        reward_list.append(x['reward'])
        hindsight_correctness_list.append(bool(x.get('hindsight_correctness', x.get('correctness', float(x['reward']) > 0))))
    input_ids_lm, _, start_pos, drop_num = uni_prompting((prompt_list, response_list))
    _, L = input_ids_lm.shape
    L0 = start_pos
    L1 = L - L0
    post_num = config.training.post_num
    for x in dataset_load:
        if 'step_map' not in x.keys() or len(x['step_map']) == 0:
            step_map_list.append([j for j in range(L1)])
        else:
            step_map_i = x['step_map']
            if len(step_map_i) > L1:
                step_map_i = step_map_i[:L1]
            else:
                step_map_i = step_map_i + [max(step_map_i) + 1] * (L1 - len(step_map_i))
            step_map_list.append(step_map_i)

    def collapse_k_unique(lst, k):
        if k <= 0:
            raise ValueError('k must be > 0')
        uniq = sorted(set(lst))
        mapping = {}
        n = len(uniq)
        for idx, val in enumerate(uniq):
            group = idx // k
            end_idx = min((group + 1) * k - 1, n - 1)
            rep = uniq[end_idx]
            mapping[val] = rep
        return [mapping[x] for x in lst]

    def make_basic_block_attention(N, start_pos, block_size):
        B = 1
        L0 = start_pos
        L1 = (N - L0) // 2
        assert L0 + 2 * L1 == N
        bias = torch.full((B, 1, N, N), False, dtype=torch.bool)
        rows = torch.arange(L0 + L1, L0 + 2 * L1)
        rows_token = torch.arange(L0, L0 + L1)
        for bi in range((L1 + block_size - 1) // block_size):
            left_end = L0 + min(bi * block_size, L1)
            right_start = L0 + L1 + (left_end - L0)
            i_start = bi * block_size
            i_end = min((bi + 1) * block_size, L1)
            block_rows = rows[i_start:i_end]
            bias[:, :, block_rows.unsqueeze(-1), 0:left_end] = 1
            bias[:, :, block_rows.unsqueeze(-1), right_start:right_start + block_size] = 1
            block_rows = rows_token[i_start:i_end]
            left_end = L0 + min((bi + 1) * block_size, L1)
            bias[:, :, block_rows.unsqueeze(-1), 0:left_end] = 1
        if L0 > 0:
            num_blocks_pre = (L0 + block_size - 1) // block_size
            for bi in range(num_blocks_pre):
                row_end = max(L0 - bi * block_size, 0)
                row_start = max(L0 - (bi + 1) * block_size, 0)
                if row_end > row_start:
                    block_rows = torch.arange(row_start, row_end)
                    bias[:, :, block_rows.unsqueeze(-1), 0:row_end] = 1
        return bias
    teacher_block_size = _teacher_block_size(config)
    basic_block_attention = make_basic_block_attention(L0 + 2 * L1, start_pos, config.training.block_size)
    basic_block_attention = basic_block_attention.cpu()
    teacher_basic_block_attention = make_basic_block_attention(L0 + 2 * L1, start_pos, teacher_block_size)
    teacher_basic_block_attention = teacher_basic_block_attention.cpu()

    def process_pad(attn, input_ids):
        N = L0 + 2 * L1
        device = input_ids.device
        cols = torch.arange(N, device=device)
        key_mask = (cols < start_pos).unsqueeze(0) & (input_ids == pad_id)
        attn.masked_fill_(key_mask[:, None, None, :], 0)
        A = attn[:, 0]
        bad = (A.sum(dim=-1) == 0) & (torch.arange(A.size(1), device=A.device).unsqueeze(0) < start_pos)
        b, r = bad.nonzero(as_tuple=True)
        A[b, r, :] = 0
        A[b, r, r] = 1
        attn = attn.bool()
        return attn

    def one_round_vectorized(input_ids_b, step_map_b, L0, L1, block_size, mask_id):
        device = input_ids_b.device
        NB = (L1 + block_size - 1) // block_size
        step_pad = torch.full((NB * block_size,), -1, dtype=torch.long, device=device)
        step_pad[:L1] = step_map_b
        step_blk = step_pad.view(NB, block_size)
        valid = step_blk.ge(0)
        big = torch.iinfo(step_blk.dtype).max
        tmp = step_blk.masked_fill(~valid, big)
        min_vals, _ = tmp.min(dim=1, keepdim=True)
        pmask_blk = step_blk.eq(min_vals) & valid
        if not pmask_blk.any():
            return (None, None, step_map_b, False)
        ge_mask_blk = step_blk.ge(min_vals) & valid
        pmask_tail = pmask_blk.view(-1)[:L1]
        ge_mask_tail = ge_mask_blk.view(-1)[:L1]
        pmask_b = torch.zeros(L0 + L1, dtype=torch.bool, device=device)
        pmask_b[L0:] = pmask_tail
        tail = input_ids_b[L0:L0 + L1].clone()
        tail[ge_mask_tail] = mask_id
        extended_input_ids_b = torch.empty(L0 + L1 + L1, dtype=input_ids_b.dtype, device=device)
        extended_input_ids_b[:L0 + L1] = input_ids_b
        extended_input_ids_b[L0 + L1:] = tail
        new_step_map_b = step_map_b.clone()
        new_step_map_b[pmask_tail] = -1
        return (extended_input_ids_b, pmask_b, new_step_map_b, True)

    def collect_training_data(input_ids, step_map_list, reward, hindsight_correctness):
        B, L = input_ids.shape
        block_size = config.training.block_size
        lower = config.training.lower_p
        upper = config.training.upper_p
        for b in range(B):
            step_map_i = step_map_list[b]
            for j in range(int((L1 - 1) / block_size) + 1):
                start = j * block_size
                end = min(L1, (j + 1) * block_size)
                step_map_list[b][start:end] = collapse_k_unique(step_map_i[start:end], config.training.shrink)
        step_map = torch.as_tensor(step_map_list, dtype=torch.long)
        assert step_map.shape[1] == L1
        extended_input_ids_list, pmask_list, reward_list, hindsight_gate_list = ([], [], [], [])
        for b in range(B):
            step_b = step_map[b]
            per_prompt_ext, per_prompt_pm = ([], [])
            while True:
                out = one_round_vectorized(input_ids_b=input_ids[b], step_map_b=step_b, L0=L0, L1=L1, block_size=block_size, mask_id=mask_id)
                extended_b, pmask_b, step_b, has_any = out
                if not has_any:
                    break
                per_prompt_ext.append(extended_b)
                per_prompt_pm.append(pmask_b)
            if not per_prompt_ext:
                continue
            else:
                ext_b, pm_b = _combine_rounds_one_state_per_block(per_prompt_ext, per_prompt_pm, input_ids[b], L0, L1, block_size)
                extended_input_ids_list.append(ext_b)
                pmask_list.append(pm_b)
                reward_list.append(reward[b])
                hindsight_gate_list.append(hindsight_correctness[b])
        extended_input_ids = torch.stack(extended_input_ids_list, dim=0)
        p_mask = torch.stack(pmask_list, dim=0).to(torch.bool)
        pad_resp = (extended_input_ids[:, :L] == pad_id) & p_mask
        if post_num is not None:
            cum_pad = torch.cumsum(pad_resp.int(), dim=1)
            p_mask &= ~(pad_resp & (cum_pad > post_num))
        labels = extended_input_ids[:, :L].clone()
        idx = torch.arange(L).unsqueeze(0).expand(extended_input_ids.shape[0], -1)
        valid = (idx >= start_pos) | extended_input_ids[:, :L].ne(pad_id)
        tok_idx = valid.long().cumsum(dim=-1) - 1
        tok_idx = tok_idx.masked_fill(~valid, 1)
        tok_idx_resp = tok_idx[:, start_pos:]
        tok_idx_ext = torch.cat([tok_idx, tok_idx_resp], dim=1)
        keep = p_mask.view(p_mask.size(0), -1).any(dim=1)
        idx = keep.nonzero(as_tuple=True)[0]
        extended_input_ids = extended_input_ids[idx]
        p_mask = p_mask[idx]
        tok_idx_ext = tok_idx_ext[idx]
        labels = labels[idx]
        reward_list = [reward_list[i] for i in idx.tolist()]
        hindsight_gate_list = [hindsight_gate_list[i] for i in idx.tolist()]
        return (extended_input_ids, p_mask, tok_idx_ext, labels, reward_list, hindsight_gate_list)
    extended_input_ids, p_mask, tok_idx_ext, labels, rewards, hindsight_correctness = collect_training_data(input_ids_lm, step_map_list, reward_list, hindsight_correctness_list)

    def simple_collate(batch):
        idx, extended_input_ids, p_mask, tok_idx_ext, labels, reward, hindsight_correctness = zip(*batch)
        return {'ids': torch.tensor(idx), 'extended_input_ids': torch.stack(extended_input_ids), 'p_mask': torch.stack(p_mask), 'tok_idx_ext': torch.stack(tok_idx_ext), 'labels': torch.stack(labels), 'reward': reward, 'hindsight_correctness': hindsight_correctness}
    dataset_lm = TrainDataset(extended_input_ids, p_mask, tok_idx_ext, labels, rewards, hindsight_correctness)
    logger.info(f'  Num training rows (after expand+filter) = {len(dataset_lm)} (from {input_ids_lm.shape[0]} responses)')
    num_train_epochs = config.training.num_train_epochs
    train_dataloader_lm = DataLoader(dataset_lm, batch_size=config.training.batch_size_lm, sampler=None, shuffle=True, collate_fn=simple_collate, num_workers=0)
    train_dataloader_lm = accelerator.prepare(train_dataloader_lm)
    if len(train_dataloader_lm) < accelerator.gradient_accumulation_steps:
        print(f'Number of batches ({len(train_dataloader_lm)}) is less than gradient accumulation steps ({accelerator.gradient_accumulation_steps}). Please reduce gradient accumulation steps or increase the number of training samples.')
    import torch.nn.functional as F

    def forward_process(extended_input_ids, p_mask, tok_idx_ext, labels, adv, logp_old_tok, hindsight_correctness=None):
        adv = torch.as_tensor(adv, device=extended_input_ids.device).detach()
        B, _L = p_mask.shape
        device = extended_input_ids.device
        attention_mask = basic_block_attention.clone()
        attention_mask = attention_mask.repeat_interleave(B, dim=0).to(device)
        attention_mask = process_pad(attention_mask, extended_input_ids)
        full_logits = model(input_ids=extended_input_ids, attention_mask=attention_mask, position_ids=tok_idx_ext).logits
        logits = torch.cat([full_logits[:, :L0, :], full_logits[:, L0 + L1:, :]], dim=1)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        with torch.no_grad():
            extended_input_ids_sampled = extended_input_ids.clone()
            masked_input_tokens = extended_input_ids_sampled == tokenizer.mask_token_id
            x0_repeated = torch.cat([extended_input_ids[:, :L0 + L1], extended_input_ids[:, L0:L0 + L1]], dim=1)
            extended_input_ids_sampled[masked_input_tokens] = x0_repeated[masked_input_tokens]
            causal_attention_mask_bool = teacher_basic_block_attention.clone()
            causal_attention_mask_bool = causal_attention_mask_bool.repeat_interleave(B, dim=0).to(device)
            causal_attention_mask_bool = process_pad(causal_attention_mask_bool, extended_input_ids)
            prompt_and_x0_mask = torch.ones(1, 1, L0 + L1, L0 + L1, dtype=torch.bool, device=device)
            prompt_and_x0_mask = torch.tril(prompt_and_x0_mask, diagonal=0).repeat(B, 1, 1, 1)
            xt_mask = torch.ones(1, 1, L1, L1, dtype=torch.bool, device=device)
            xt_mask = torch.tril(xt_mask, diagonal=0).repeat(B, 1, 1, 1)
            causal_attention_mask_bool[:, :, :L0 + L1, :L0 + L1] &= prompt_and_x0_mask
            causal_attention_mask_bool[:, :, L0 + L1:, L0 + L1:] &= xt_mask
            causal_attention_mask = torch.full(causal_attention_mask_bool.shape, -1e309, dtype=torch.bfloat16, device=device)
            causal_attention_mask[causal_attention_mask_bool] = 0.0
            block_size = teacher_block_size
            logits_teacher_full = teacher_model(input_ids=extended_input_ids_sampled, attention_mask=causal_attention_mask.bfloat16(), position_ids=tok_idx_ext).logits
            logits_teacher = torch.cat([logits_teacher_full[:, :L0, :], logits_teacher_full[:, L0 + L1:, :]], dim=1)
            if L1 >= block_size:
                replace_x0_indices = torch.arange(start=L0 + block_size - 1, end=L0 + L1, step=block_size, device=logits_teacher.device)
                logits_teacher[:, replace_x0_indices] = logits_teacher_full[:, replace_x0_indices]
            logits_teacher = logits_teacher.roll(dims=1, shifts=1)
            _lt = logits_teacher.float()
            teacher_logprobs = F.log_softmax(_lt, dim=-1)
        hindsight_stats = {}
        teacher_logprobs = _apply_aligned_hindsight_correction(teacher_logprobs=teacher_logprobs, student_logprobs=log_probs, teacher_model=teacher_model, hindsight_scorer_model=hindsight_scorer_model, extended_input_ids=extended_input_ids, tok_idx_ext=tok_idx_ext, L0=L0, L1=L1, tokenizer=tokenizer, config=config, rewards=adv, hindsight_correctness=hindsight_correctness, stats=hindsight_stats)
        kl_div = _compute_token_divergence(teacher_logprobs, log_probs, config)
        prompt = extended_input_ids[:, :L0]
        prompt_response = extended_input_ids[:, :L0 + L1]
        response_noised = extended_input_ids[:, L0 + L1:]
        prompt_mask = torch.cat([torch.zeros_like(prompt), torch.ones_like(response_noised)], dim=1).bool()
        im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
        is_im_end = prompt_response.eq(im_end_id)
        is_response_im_end = is_im_end & prompt_mask
        im_end_cumsum = is_response_im_end.cumsum(dim=1)
        im_end_shifted = F.pad(im_end_cumsum[:, :-1], (1, 0))
        im_end_mask = im_end_shifted.eq(0)
        if getattr(config.training, 'exclude_im_end', False):
            im_end_mask = im_end_mask & ~is_response_im_end
        prompt_noised_response = torch.cat([prompt, response_noised], dim=1)
        pad_mask_t = prompt_noised_response.ne(tokenizer.pad_token_id)
        masked_token_mask = prompt_noised_response.eq(tokenizer.mask_token_id)
        response_mask = prompt_mask & pad_mask_t & im_end_mask
        loss_mask = response_mask & masked_token_mask
        kl_div_mask = kl_div * loss_mask
        kd_loss_unreduced = kl_div_mask.sum(dim=1) / loss_mask.sum(dim=1).clamp_min(1.0)
        loss_unreduced = kd_loss_unreduced
        num_response_tokens = response_mask.sum(dim=1)
        num_masked_tokens = (response_mask & masked_token_mask).sum(dim=1)
        frac_masked = num_masked_tokens.float() / num_response_tokens.float().clamp(min=1)
        loss_by_masking_rate = {}
        for frac, loss_i in zip(frac_masked, loss_unreduced):
            metric_name = f'train/loss_masked_frac_{frac:.1f}'
            num_rate, total_loss = loss_by_masking_rate.get(metric_name, (0, 0))
            loss_by_masking_rate[metric_name] = (num_rate + 1, total_loss + loss_i.item())
        _add_hindsight_metrics(loss_by_masking_rate, hindsight_stats)
        loss = loss_unreduced.mean()
        num_forward_tokens = response_mask.sum().item()
        return (loss, loss_by_masking_rate, num_forward_tokens)
    first_epoch = 0
    data_time_m = AverageMeter()
    end = time.time()
    if wandb_enabled and len(rewards) > 0:
        reward_arr = np.asarray(rewards, dtype=np.float32)
        accelerator.log({'train/reward_mean': float(reward_arr.mean()), 'train/reward_std': float(reward_arr.std()), 'train/reward_min': float(reward_arr.min()), 'train/reward_max': float(reward_arr.max()), 'train/current_epoch': current_epoch})
    from tqdm.auto import tqdm
    hindsight_step_totals = {}
    stepped = False
    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        progress_bar = tqdm(train_dataloader_lm, desc=f'Epoch {epoch + 1}/{num_train_epochs}', disable=not accelerator.is_local_main_process, dynamic_ncols=True, leave=True)
        metrics = {}
        avg_loss = AverageMeter()
        for step, batch in enumerate(progress_bar):
            data_time_m.update(time.time() - end)
            extended_input_ids_b = batch['extended_input_ids'].to(accelerator.device)
            p_mask_b = batch['p_mask'].to(accelerator.device)
            tok_idx_ext_b = batch['tok_idx_ext'].to(accelerator.device)
            labels_b = batch['labels'].to(accelerator.device)
            reward_b = batch['reward']
            hindsight_correctness_b = batch['hindsight_correctness']
            old_lp = dataset_lm.logp_old_tok[batch['ids'].cpu()].to(accelerator.device)
            _fwd_fn = forward_process
            loss_lm, loss_metrics, batch_loss_tokens = _fwd_fn(extended_input_ids=extended_input_ids_b, p_mask=p_mask_b, tok_idx_ext=tok_idx_ext_b, labels=labels_b, adv=reward_b, logp_old_tok=old_lp, hindsight_correctness=hindsight_correctness_b)
            _batch_loss_tokens_tensor = torch.tensor(batch_loss_tokens, device=accelerator.device)
            torch.distributed.all_reduce(_batch_loss_tokens_tensor, op=torch.distributed.ReduceOp.SUM)
            cumulative_loss_tokens += _batch_loss_tokens_tensor.item()
            for loss_metric_name, (loss_metric_ct, loss_metric_value) in loss_metrics.items():
                if loss_metric_name not in metrics:
                    metrics[loss_metric_name] = AverageMeter()
                loss_metric_value = loss_metric_value / loss_metric_ct
                metrics[loss_metric_name].update(loss_metric_value, n=loss_metric_ct)
            avg_loss.update(loss_lm.item(), n=len(extended_input_ids_b))
            with accelerator.accumulate(model):
                accelerator.backward(loss_lm)
                if accelerator.sync_gradients:
                    stepped = True
                    _clip_val = config.training.max_grad_norm if config.training.max_grad_norm is not None else 1e309
                    grad_norm_before_clip = accelerator.clip_grad_norm_(model.parameters(), _clip_val)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    float2tensor = lambda x: torch.tensor(x, device=accelerator.device)
                    all_sums = accelerator.gather(float2tensor(avg_loss.sum))
                    all_counts = accelerator.gather(float2tensor(avg_loss.count))
                    all_avg = all_sums.sum() / all_counts.sum().clamp_min(1)
                    all_possible_keys = sorted({f'train/loss_masked_frac_{round(f, 1)}' for f in [i / 10 for i in range(11)]} | set(_HINDSIGHT_SUM_METRIC_NAMES) | {'train/loss_causal'})
                    local_flags = torch.tensor([1.0 if k in metrics else 0.0 for k in all_possible_keys], device=accelerator.device)
                    global_flags = accelerator.gather(local_flags)
                    if global_flags.ndim > 1:
                        global_flags = global_flags.sum(dim=0)
                    all_metrics_keys = [k for k, f in zip(all_possible_keys, global_flags) if f > 0]
                    all_metrics = {}
                    for key in all_metrics_keys:
                        all_sums = accelerator.gather(float2tensor(metrics[key].sum if key in metrics else 0.0))
                        all_counts = accelerator.gather(float2tensor(metrics[key].count if key in metrics else 0))
                        if key in _HINDSIGHT_SUM_METRIC_NAMES:
                            all_metrics[key] = all_sums.sum()
                        elif all_counts.sum() > 0:
                            all_metrics[key] = all_sums.sum() / all_counts.sum()
                    logged_metrics = {'train/loss': all_avg.item(), 'train/lr': float(lr_scheduler.get_last_lr()[0]), 'train/epoch': float(epoch + 1), **all_metrics}
                    logged_metrics['train/grad_norm'] = grad_norm_before_clip.item() if hasattr(grad_norm_before_clip, 'item') else float(grad_norm_before_clip)
                    logged_metrics['train/num_training_samples'] = input_ids_lm.shape[0]
                    logged_metrics['train/cumulative_loss_tokens'] = cumulative_loss_tokens
                    logged_metrics['train/student_cumulative_tflops'] = 6 * _student_total_params * cumulative_loss_tokens / 1e+18
                    logged_metrics['train/teacher_cumulative_tflops'] = 2 * _teacher_total_params * cumulative_loss_tokens / 1e+18
                    _rollout_token_file = os.path.join(config.experiment.project, 'temp_data', 'cumulative_rollout_tokens.txt')
                    _cumulative_rollout_tokens = 0
                    if os.path.exists(_rollout_token_file):
                        with open(_rollout_token_file) as _f:
                            _cumulative_rollout_tokens = int(_f.read().strip())
                    logged_metrics['train/rollout_cumulative_tokens'] = _cumulative_rollout_tokens
                    logged_metrics['train/rollout_cumulative_tflops'] = 2 * _student_total_params * _cumulative_rollout_tokens / 1e+18
                    logged_metrics['train/total_cumulative_tflops'] = logged_metrics['train/student_cumulative_tflops'] + logged_metrics['train/teacher_cumulative_tflops']
                    _gpu_hours_file = os.path.join(config.experiment.project, 'temp_data', 'cumulative_gpu_hours.txt')
                    if os.path.exists(_gpu_hours_file):
                        with open(_gpu_hours_file) as _ghf:
                            logged_metrics['train/cumulative_gpu_hours'] = float(_ghf.read().strip())
                    else:
                        logged_metrics['train/cumulative_gpu_hours'] = 0.0
                    logged_metrics['train/current_epoch'] = current_epoch
                    _completed_gpu_hour_steps = max(int(current_epoch) - 1, 0)
                    logged_metrics['train/gpu_hours_completed_steps_for_avg'] = _completed_gpu_hour_steps
                    logged_metrics['train/avg_gpu_hours_per_completed_step'] = logged_metrics['train/cumulative_gpu_hours'] / _completed_gpu_hour_steps if _completed_gpu_hour_steps > 0 else 0.0
                    _add_hindsight_derived_metrics(logged_metrics, hindsight_step_totals)
                    if wandb_enabled:
                        accelerator.log(logged_metrics)
                        metrics = {}
                    avg_loss.reset()
                    torch.cuda.empty_cache()
            end = time.time()
    _log_hindsight_summary(accelerator, hindsight_step_totals)
    if not stepped:
        logger.warning('Training ended with no optimizer step taken.')
    token_count_file = os.path.join(config.experiment.project, 'temp_data', 'cumulative_loss_tokens.txt')
    os.makedirs(os.path.dirname(token_count_file), exist_ok=True)
    with open(token_count_file, 'w') as f:
        f.write(str(cumulative_loss_tokens))
    state['cumulative_loss_tokens'] = cumulative_loss_tokens
    accelerator.wait_for_everyone()
    save_named_epoch, save_full_state = _checkpoint_save_policy(config)
    save_checkpoint(model, tokenizer, config, accelerator, config.model.optimized_name, save_training_state_flag=save_full_state, lr_scheduler=lr_scheduler if save_full_state else None)
    if save_named_epoch:
        save_checkpoint(model, tokenizer, config, accelerator, f'epoch-{config.experiment.current_epoch}', save_training_state_flag=False)
        prune_epoch_checkpoints(config, accelerator)
