"""Create an untrained diffusion student from a local dense Qwen3 checkpoint."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dopd.mask_token_utils import ensure_diffusion_mask_token
    model = AutoModelForCausalLM.from_pretrained(args.source, torch_dtype=torch.bfloat16, trust_remote_code=False)
    if model.config.model_type != 'qwen3':
        raise ValueError('Expected a dense Qwen3 autoregressive model')
    tokenizer = AutoTokenizer.from_pretrained(args.source, trust_remote_code=False)
    ensure_diffusion_mask_token(tokenizer)
    if tokenizer.mask_token_id >= model.config.vocab_size:
        raise ValueError('The checkpoint must reserve a vocabulary row for the diffusion mask token')
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    import json
    path = destination / 'config.json'
    config = json.loads(path.read_text())
    config.update(model_type='a2d-qwen3', architectures=['A2DQwen3LMHeadModel'])
    config.pop('auto_map', None)
    path.write_text(json.dumps(config, indent=2) + '\n')
    print(f'Saved diffusion student with unchanged weights to {destination}')


if __name__ == '__main__':
    main()
