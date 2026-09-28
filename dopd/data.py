"""External data and task-specific correctness callbacks; no bundled datasets."""
import importlib
import json
from pathlib import Path


def load_records(path):
    path = Path(path)
    with path.open(encoding='utf-8') as handle:
        records = [json.loads(line) for line in handle if line.strip()] if path.suffix == '.jsonl' else json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError('Training data must be a nonempty JSON array or JSONL file')
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get('prompt'), str):
            raise ValueError(f'Record {index} must contain a string prompt')
    return records


def load_correctness_fn(spec):
    if spec is None:
        return None
    module, name = spec.split(':', 1)
    fn = getattr(importlib.import_module(module), name)
    if not callable(fn):
        raise ValueError('data.correctness_fn must identify a callable')
    return fn


def format_prompt(record, tokenizer, cfg):
    if not cfg.data.apply_chat_template:
        return record['prompt']
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': record['prompt']}], tokenize=False,
        add_generation_prompt=True, enable_thinking=cfg.data.enable_thinking,
    )
