import atexit
from collections import Counter
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
import torch
from torch import nn
from contextlib import nullcontext
from dopd.jetengine.config import Config
from dopd.jetengine.sampling_params import SamplingParams
from dopd.jetengine.engine.sequence import Sequence, RunType
from dopd.jetengine.engine.scheduler import Scheduler
from dopd.jetengine.engine.model_runner import ModelRunner
from dopd.jetengine.utils.loader import load_from_hf_model

class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context('spawn')
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True, trust_remote_code=True)
        config.eos = self.tokenizer.eos_token_id
        config.mask_token_id = self.tokenizer.mask_token_id if self.tokenizer.mask_token_id is not None else 151669
        assert config.mask_token_id is not None, 'Model tokenizer must have a mask_token_id or pad_token_id'
        self.config = config
        self.scheduler = Scheduler(config)
        self.scheduler.consistent_sampling_params = False
        atexit.register(self.exit)

    def offload_parameters(self, include_buffers: bool=False):
        """
        Replace all parameter (and buffer) storages with meta tensors.
        Keeps shapes/dtypes, frees GPU/CPU memory.
        """

        def offload_parameters_keep_buffers(model: torch.nn.Module):
            """
            Move *parameters* to meta to free memory while keeping buffers unchanged.
            Works for any module tree.
            """
            saved_buffers = []
            for mod in model.modules():
                for bname, buf in list(mod._buffers.items()):
                    if buf is not None:
                        saved_buffers.append((mod, bname, buf))
            model.to_empty(device=torch.device('meta'))
            for mod, bname, buf in saved_buffers:
                mod._buffers[bname] = buf
            torch.cuda.empty_cache()
        if include_buffers:
            self.model_runner.model.to_empty(device=torch.device('meta'))
        else:
            offload_parameters_keep_buffers(self.model_runner.model)
        print('Successfully cleaned old parameters (buffers kept).' if not include_buffers else 'Successfully cleaned old parameters and buffers.')

    def reload_parameters(self, hf_model: nn.Module):
        load_from_hf_model(self.model_runner.model, hf_model=hf_model)

    def sleep(self):
        """Free GPU memory (parameters + KV cache + CUDA graphs) for training."""
        import gc
        mr = self.model_runner
        if hasattr(mr, 'graphs'):
            for graph in mr.graphs.values():
                graph.reset()
            del mr.graphs
        if hasattr(mr, 'graph_pool'):
            del mr.graph_pool
        for module in mr.model.modules():
            if hasattr(module, 'k_cache'):
                module.k_cache = None
            if hasattr(module, 'v_cache'):
                module.v_cache = None
        if hasattr(mr, 'kv_cache'):
            del mr.kv_cache
        del mr.model
        mr.model = None
        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f'JetEngine sleep: freed all GPU memory. Free: {free / 1000000000.0:.1f} GB / {total / 1000000000.0:.1f} GB')

    def wake_up(self, hf_model: nn.Module):
        """Recreate model from HF weights and re-allocate KV cache after training."""
        mr = self.model_runner
        hf_config = self.config.hf_config
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device('cuda')
        from dopd.jetengine.models.sdar import SDARForCausalLM
        if 'sdar' in hf_config.model_type or hf_config.model_type in ('qwen3', 'qwen2', 'a2d-qwen3'):
            mr.model = SDARForCausalLM(hf_config)
        else:
            raise ValueError(f'Unsupported model type for wake_up: {hf_config.model_type}')
        torch.set_default_device('cpu')
        torch.set_default_dtype(default_dtype)
        load_from_hf_model(mr.model, hf_model=hf_model)
        mr.allocate_kv_cache()
        if not mr.enforce_eager:
            mr.capture_cudagraph()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f'JetEngine wake_up: reloaded. Free: {free / 1000000000.0:.1f} GB / {total / 1000000000.0:.1f} GB')

    def wake_up_from_path(self, checkpoint_path: str):
        """Recreate model from safetensors checkpoint and re-allocate KV cache.

        Uses JetEngine's own load_model (reads safetensors directly) to bypass
        DeepSpeed ZeRO-3 hooks that would intercept from_pretrained.
        """
        from dopd.jetengine.utils.loader import load_model as je_load_model
        mr = self.model_runner
        hf_config = self.config.hf_config
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device('cuda')
        from dopd.jetengine.models.sdar import SDARForCausalLM
        if 'sdar' in hf_config.model_type or hf_config.model_type in ('qwen3', 'qwen2', 'a2d-qwen3'):
            mr.model = SDARForCausalLM(hf_config)
        else:
            raise ValueError(f'Unsupported model type for wake_up: {hf_config.model_type}')
        je_load_model(mr.model, checkpoint_path, tp_size=1, tp_rank=0)
        mr.model.cuda()
        mr.warmup_model()
        mr.allocate_kv_cache()
        if not mr.enforce_eager:
            mr.capture_cudagraph()
        torch.set_default_device('cpu')
        torch.set_default_dtype(default_dtype)
        self.scheduler = Scheduler(self.config)
        self.scheduler.consistent_sampling_params = False
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f'JetEngine wake_up_from_path: reloaded from {checkpoint_path}. Free: {free / 1000000000.0:.1f} GB / {total / 1000000000.0:.1f} GB')

    def exit(self):
        self.model_runner.call('exit')
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if isinstance(prompt, list):
            if self.tokenizer.pad_token_id in prompt:
                start = prompt.index(self.tokenizer.pad_token_id) + 1
                prompt = prompt[start:]
        seq = Sequence(prompt, self.config.mask_token_id, sampling_params)
        seq.eos_token_id = self.tokenizer.eos_token_id
        self.scheduler.add(seq)

    def step(self):
        scheduled_seqs, run_type = self.scheduler.schedule()
        if scheduled_seqs is None:
            running = getattr(self.scheduler, 'running', [])
            if running:
                block_manager = self.scheduler.block_manager
                needed = [seq.num_new_blocks_needed(block_manager.block_size) for seq in running]
                statuses = dict(Counter((seq.status.name for seq in running)))
                raise RuntimeError(f'Scheduler stalled with active sequences but no schedulable batch: running={len(running)}, free_blocks={len(block_manager.free_block_ids)}, needed_blocks_min={min(needed)}, needed_blocks_max={max(needed)}, statuses={statuses}')
            return ([], 0)
        logits = self.model_runner.call('run', scheduled_seqs, run_type)
        self.scheduler.postprocess(scheduled_seqs, logits, run_type)
        finished_outputs = [(seq.seq_id, seq.completion_token_ids, seq.first_unmask_steps) for seq in scheduled_seqs if seq.is_finished]
        num_tokens = [self.scheduler.running[i].num_to_transfer if hasattr(self.scheduler.running[i], 'num_to_transfer') else 0 for i in range(len(self.scheduler.running))]
        return (finished_outputs, sum(num_tokens))

    def is_finished(self):
        return self.scheduler.is_finished()

    def _clean_token_ids(self, token_ids):
        try:
            token_ids = list(token_ids)
        except Exception:
            token_ids = [token_ids]
        vocab_size = getattr(self.tokenizer, 'vocab_size', None)
        special_ids = set(getattr(self.tokenizer, 'all_special_ids', []) or [])
        mask_id = getattr(self.config, 'mask_token_id', None)
        cleaned = []
        for t in token_ids:
            if t is None or t < 0 or t == mask_id or (t >= vocab_size):
                if t not in special_ids:
                    cleaned.append(0)
                    continue
            cleaned.append(t)
        return cleaned

    def _safe_decode(self, token_ids):
        ids = self._clean_token_ids(token_ids)
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def generate(self, prompts: list[str] | list[list[int]], sampling_params: SamplingParams | list[SamplingParams], use_tqdm: bool=True) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc='Generating', dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
            self.scheduler.consistent_sampling_params = True
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        total_generated_tokens = 0
        start_time = perf_counter()
        while not self.is_finished():
            output, num_processed = self.step()
            total_generated_tokens += num_processed
            throughput = total_generated_tokens / (perf_counter() - start_time)
            if use_tqdm:
                pbar.set_postfix({'Throughput': f'{int(throughput)} tok/s'})
            for seq_id, token_ids, unmask_times in output:
                outputs[seq_id] = {'token_ids': token_ids, 'unmask_times': unmask_times}
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{'text': self._safe_decode(item['token_ids']), 'token_ids': self._clean_token_ids(item['token_ids']), 'first_unmask_times': item['unmask_times']} for item in outputs]
        if use_tqdm:
            pbar.close()
        return outputs

    def generate_streaming(self, prompts: list[str] | list[list[int]], sampling_params: SamplingParams | list[SamplingParams], max_active: int | None=None, use_tqdm: bool=True) -> list[str]:
        """
        Stream prompts through the engine while keeping up to `max_active` sequences running.
        As sequences finish, new prompts are added from the pending list to maximize GPU utilization.
        """
        total = len(prompts)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * total
            self.scheduler.consistent_sampling_params = True
        if max_active is None:
            max_active = getattr(self.scheduler, 'max_num_seqs', 32)
        if use_tqdm:
            pbar = tqdm(total=total, desc='Generating', dynamic_ncols=True)
        outputs: dict[int, list[int]] = {}
        pending_idx = 0
        initial = min(max_active, total)
        for i in range(initial):
            self.add_request(prompts[i], sampling_params[i])
        pending_idx = initial
        total_generated_tokens = 0
        start_time = perf_counter()
        while not self.is_finished() or pending_idx < total:
            running = getattr(self.scheduler, 'running', [])
            deficit = max_active - len(running)
            while deficit > 0 and pending_idx < total:
                self.add_request(prompts[pending_idx], sampling_params[pending_idx])
                pending_idx += 1
                deficit -= 1
            output, num_processed = self.step()
            total_generated_tokens += num_processed
            if use_tqdm:
                throughput = total_generated_tokens / (perf_counter() - start_time + 1e-06)
                pbar.set_postfix({'Throughput': f'{int(throughput)} tok/s'})
                pbar.update(len(output))
            for seq_id, token_ids, unmask_times in output:
                outputs[seq_id] = {'token_ids': token_ids, 'unmask_times': unmask_times}
        outputs_list = [outputs[seq_id] for seq_id in sorted(outputs)]
        results = [{'text': self._safe_decode(item['token_ids']), 'token_ids': self._clean_token_ids(item['token_ids']), 'first_unmask_times': item['unmask_times']} for item in outputs_list]
        if use_tqdm:
            pbar.close()
        return results
