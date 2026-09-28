"""Verify runtime dependencies and small kernels, without model/data downloads."""
import argparse
from importlib.metadata import version
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-cuda', action='store_true', help='Check imports/config only; no GPU validation')
    parser.add_argument('--skip-build', action='store_true', help='Skip DeepSpeed CPUAdam native extension validation')
    args = parser.parse_args()
    expected = {'torch': '2.6.0', 'triton': '3.2.0', 'transformers': '4.52.4',
                'accelerate': '1.11.0', 'deepspeed': '0.18.0', 'flash-attn': '2.7.4.post1',
                'liger-kernel': '0.7.0'}
    for package, required in expected.items():
        actual = version(package)
        if actual.split('+')[0] != required:
            raise RuntimeError(f'{package}: expected {required}, found {actual}')
        print(f'{package}: {actual}', flush=True)
    import torch
    from dopd import trainer
    from dopd.config import load_config, training_config
    config = load_config(Path(__file__).parent / 'configs/train.yaml')
    training_config(config, 1)
    print('Training imports and configuration: OK', flush=True)

    if not args.skip_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Check the NVIDIA driver and GPU visibility.')
        if torch.version.cuda != '12.4':
            raise RuntimeError(f'Expected a CUDA 12.4 PyTorch build, found {torch.version.cuda}')
        major, minor = torch.cuda.get_device_capability()
        if major not in (8, 9):
            raise RuntimeError('This dependency stack targets Ampere, Ada or Hopper GPUs.')
        print(f'GPU: {torch.cuda.get_device_name()} (sm_{major}{minor})', flush=True)
        from flash_attn import flash_attn_func, flash_attn_with_kvcache
        from dopd.jetengine.llm import LLM
        from dopd.jetengine.layers.attention import store_kvcache
        torch.manual_seed(0)
        q, k, v = [torch.randn(1, 8, 2, 64, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
        actual = flash_attn_func(q, k, v, causal=False)
        expected_attention = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float(),
        ).transpose(1, 2)
        torch.testing.assert_close(actual.float(), expected_attention, atol=0.025, rtol=0.025)
        key, value = k[0].contiguous(), v[0].contiguous()
        key_cache = torch.zeros(1, 8, 2, 64, device='cuda', dtype=torch.bfloat16)
        value_cache = torch.zeros_like(key_cache)
        store_kvcache(key, value, key_cache, value_cache, torch.arange(8, device='cuda'))
        torch.testing.assert_close(key_cache[0], key, atol=0, rtol=0)
        torch.testing.assert_close(value_cache[0], value, atol=0, rtol=0)
        query = q[:, -1:].contiguous()
        cached_attention = flash_attn_with_kvcache(
            query, key_cache, value_cache,
            cache_seqlens=torch.tensor([8], device='cuda', dtype=torch.int32), causal=False,
        )
        reference = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float(),
        ).transpose(1, 2)
        torch.testing.assert_close(cached_attention.float(), reference, atol=0.025, rtol=0.025)
        torch.cuda.synchronize()
        from dopd.jetengine.layers.activation import SiluAndMul
        activation = torch.randn(4, 128, device='cuda', dtype=torch.bfloat16)
        expected_activation = torch.nn.functional.silu(activation[:, :64]) * activation[:, 64:]
        torch.testing.assert_close(SiluAndMul()(activation), expected_activation, atol=0.025, rtol=0.025)
        torch.cuda.synchronize()
        print('FlashAttention, Triton KV-cache and Liger activation kernels: OK', flush=True)

    if not args.skip_build:
        import deepspeed
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        parameter = torch.nn.Parameter(torch.ones(8))
        optimizer = DeepSpeedCPUAdam([parameter], lr=0.01)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        if not torch.isfinite(parameter).all() or not (parameter < 1).all():
            raise RuntimeError('DeepSpeed CPUAdam did not perform a valid parameter update')
        del optimizer
        print('DeepSpeed CPUAdam native extension: OK', flush=True)

    skipped = [name for flag, name in [(args.skip_cuda, 'CUDA'), (args.skip_build, 'CPUAdam build')] if flag]
    if skipped:
        print('Partial checks passed; skipped: ' + ', '.join(skipped))
    else:
        print('Environment checks passed (not a full training run).')


if __name__ == '__main__':
    main()
