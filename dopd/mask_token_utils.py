DIFFUSION_MASK_TOKEN = "<|MASK|>"
DIFFUSION_MASK_TOKEN_ID = 151669


def ensure_diffusion_mask_token(tokenizer, mask_token=DIFFUSION_MASK_TOKEN, mask_token_id=DIFFUSION_MASK_TOKEN_ID):
    """Make tokenizer expose the reserved diffusion mask token without resizing vocab.

    Qwen tokenizers often already contain token id 151669 but do not mark it as
    mask_token. Adding it as an additional special token records that mapping so
    JetEngine, eval, and saved checkpoints agree with training.
    """
    if tokenizer.mask_token_id == mask_token_id:
        return tokenizer
    existing_id = tokenizer.convert_tokens_to_ids(mask_token)
    if existing_id == mask_token_id:
        tokenizer.mask_token = mask_token
        return tokenizer
    if existing_id is None or existing_id == tokenizer.unk_token_id:
        added = tokenizer.add_special_tokens({"mask_token": mask_token})
        if added and len(tokenizer) - 1 != mask_token_id:
            raise ValueError(
                f"Added {mask_token} at id {len(tokenizer) - 1}, expected reserved id {mask_token_id}."
            )
    tokenizer.mask_token = mask_token
    if tokenizer.mask_token_id != mask_token_id:
        raise ValueError(
            f"{mask_token} resolved to id {tokenizer.mask_token_id}, expected {mask_token_id}."
        )
    return tokenizer


def copy_pad_embedding_to_mask(model, tokenizer=None, pad_id=None, mask_id=None, log_fn=None):
    import torch

    if tokenizer is not None:
        pad_id = tokenizer.pad_token_id if pad_id is None else pad_id
        mask_id = tokenizer.mask_token_id if mask_id is None else mask_id
    if pad_id is None or mask_id is None:
        raise ValueError("copy_pad_embedding_to_mask requires both pad_id and mask_id")
    pad_id = int(pad_id)
    mask_id = int(mask_id)
    if pad_id == mask_id:
        return []

    copied = []

    def _copy_rows(module, name):
        if module is None or not hasattr(module, "weight"):
            return
        weight = module.weight
        start = int(getattr(module, "vocab_start_idx", 0))
        end = int(getattr(module, "vocab_end_idx", weight.shape[0]))
        has_pad = start <= pad_id < end
        has_mask = start <= mask_id < end
        if not has_pad and not has_mask:
            return
        if has_pad != has_mask:
            raise ValueError(
                f"Cannot copy pad embedding to mask for tensor-parallel shard {name}: "
                f"pad_id={pad_id}, mask_id={mask_id}, shard=[{start}, {end})."
            )
        with torch.no_grad():
            weight[mask_id - start].copy_(weight[pad_id - start])
        copied.append(name)

    input_embeddings = _get_input_embeddings(model)
    _copy_rows(input_embeddings, "input_embeddings")
    output_embeddings = _get_output_embeddings(model)
    if output_embeddings is not None and output_embeddings is not input_embeddings:
        _copy_rows(output_embeddings, "output_embeddings")

    if log_fn is not None:
        if copied:
            log_fn(
                f"Copied pad embedding row {pad_id} to mask row {mask_id}: "
                + ", ".join(copied)
            )
        else:
            log_fn(
                f"Skipped pad-to-mask embedding copy on this vocab shard "
                f"(pad_id={pad_id}, mask_id={mask_id})."
            )
    return copied


def _get_input_embeddings(model):
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if embeddings is not None:
            return embeddings
    return getattr(getattr(model, "model", None), "embed_tokens", None)


def _get_output_embeddings(model):
    if hasattr(model, "get_output_embeddings"):
        embeddings = model.get_output_embeddings()
        if embeddings is not None:
            return embeddings
    return getattr(model, "lm_head", None)
