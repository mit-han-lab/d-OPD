"""Future-aware teacher correction and token-level distillation objective.

Model loading, trajectory generation and the training loop are supplied by the caller.
"""
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

_HINDSIGHT_STAT_KEYS = ('rows', 'correct_rows', 'rows_with_jobs', 'rows_with_apply', 'eligible_left_positions', 'jobs', 'applied_positions', 'candidate_tokens', 'finite_candidate_tokens', 'scored_branches', 'skipped_no_visible_blocks', 'skipped_no_future_blocks', 'skipped_no_loss_positions', 'skipped_no_candidates', 'skipped_correctness_only_rows', 'skipped_nonfinite_jobs', 'skipped_bad_normalizer_jobs')

def _reset_hindsight_stats(stats, rows=0):
    stats.clear()
    for key in _HINDSIGHT_STAT_KEYS:
        stats[key] = 0
    stats['rows'] = int(rows)

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

def _compute_token_divergence(teacher_logprobs, student_logprobs, config):
    k = int(config.training.top_k_logits)
    if k > 0:
        teacher_logprobs, indices = teacher_logprobs.topk(min(k, teacher_logprobs.size(-1)), dim=-1)
        student_logprobs = student_logprobs.gather(-1, indices)
    finite = torch.isfinite(teacher_logprobs)
    terms = torch.where(finite, teacher_logprobs.exp() * (teacher_logprobs - student_logprobs), torch.zeros_like(teacher_logprobs))
    return terms.sum(dim=-1)

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


def correct_teacher(*, config, **kwargs):
    """Correct teacher log-probabilities for a caller-supplied denoising state.

    ``config`` is MethodConfig; other arguments follow the tensor interface of
    ``_apply_aligned_hindsight_correction``. Supply a frozen AR scoring model.
    """
    return _apply_aligned_hindsight_correction(config=config.internal(), **kwargs)


def distillation_loss(teacher_logprobs, student_logprobs, config):
    """Return per-token forward KL; the caller applies the loss mask and reduction."""
    return _compute_token_divergence(teacher_logprobs, student_logprobs, config.internal())
