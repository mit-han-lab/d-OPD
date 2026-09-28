import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from omegaconf import OmegaConf
from dopd.config import load_config, training_config
from dopd.jetengine.sampling_params import SamplingParams
from dopd import trainer

CONFIG = Path(__file__).parents[1] / 'configs/train.yaml'


class ConfigTests(unittest.TestCase):
    def test_denoise_tracks_block_size(self):
        for n in (2, 4, 8, 16):
            self.assertEqual(load_config(CONFIG, [f'block_size={n}']).denoise_step, n)
            self.assertEqual(SamplingParams(block_length=n).denoising_steps, n)

    def test_explicit_and_null_denoise(self):
        self.assertEqual(load_config(CONFIG, ['block_size=8', 'denoise_step=3']).denoise_step, 3)
        self.assertEqual(load_config(CONFIG, ['block_size=8', 'denoise_step=null']).denoise_step, 8)
        for value in ('0', '-1', '2.5'):
            with self.assertRaises(ValueError):
                load_config(CONFIG, [f'denoise_step={value}'])

    def test_typos_rejected(self):
        with self.assertRaises(Exception):
            load_config(CONFIG, ['denoise_steps=5'])


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cls.model_cfg = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            head_dim=8, pad_token_id=0, eos_token_id=2, attn_implementation='eager')
        cls.teacher = Qwen3ForCausalLM(cls.model_cfg).eval()
        cls.tokenizer = SimpleNamespace(pad_token_id=0, mask_token_id=31,
                                       convert_tokens_to_ids=lambda _: 2)

    def test_cached_branch_scores_match_full_forward(self):
        sequence = torch.tensor([3, 4, 5, 6, 7, 8])
        pos = torch.arange(6)
        cache = trainer._build_aligned_ar_cache(self.teacher, sequence, pos, 0)
        candidates = torch.tensor([5, 9, 10])
        scores = trainer._score_aligned_future_branches(self.teacher, cache, sequence, pos,
            prefix_length=2, branch_pos=2, branch_token_ids=candidates,
            future_start=3, future_end=6, pad_id=0, max_chunk_size=2)
        expected = []
        with torch.no_grad():
            for candidate in candidates:
                altered = sequence.clone(); altered[2] = candidate
                lp = self.teacher(altered[None]).logits.float().log_softmax(-1)
                expected.append(lp[0, 2:5].gather(-1, sequence[3:6, None]).sum())
        torch.testing.assert_close(scores, torch.stack(expected), atol=1e-6, rtol=1e-5)
        self.assertEqual(cache.get_seq_length(), 6)

    def correction_inputs(self):
        cfg = training_config(load_config(CONFIG), 1)
        cfg.training.hindsight_top_k = 3
        x = torch.tensor([[3, 4, 5, 6, 7, 8, 31, 6, 31, 8]])
        pos = torch.tensor([[0, 1, 2, 3, 4, 5, 2, 3, 4, 5]])
        t = torch.randn(1, 6, 32).log_softmax(-1)
        s = torch.randn(1, 6, 32).log_softmax(-1)
        return dict(teacher_logprobs=t, student_logprobs=s, teacher_model=self.teacher,
                    hindsight_scorer_model=self.teacher, extended_input_ids=x, tok_idx_ext=pos,
                    L0=2, L1=4, tokenizer=self.tokenizer, config=cfg, hindsight_correctness=[True])

    def test_correction_normalized_gated_and_lambda_zero(self):
        kwargs = self.correction_inputs()
        stats = {}
        corrected = trainer._apply_aligned_hindsight_correction(**kwargs, stats=stats)
        self.assertGreater(stats['applied_positions'], 0)
        torch.testing.assert_close(corrected.exp().sum(-1), torch.ones(1, 6))
        self.assertFalse(torch.equal(corrected, kwargs['teacher_logprobs']))
        kwargs['hindsight_correctness'] = [False]
        self.assertTrue(torch.equal(trainer._apply_aligned_hindsight_correction(**kwargs), kwargs['teacher_logprobs']))
        kwargs['config'].training.hindsight_lambda = 0
        kwargs['hindsight_correctness'] = [True]
        self.assertTrue(torch.equal(trainer._apply_aligned_hindsight_correction(**kwargs), kwargs['teacher_logprobs']))

    def test_selected_forward_kl_has_student_gradient(self):
        cfg = training_config(load_config(CONFIG), 1)
        cfg.training.top_k_logits = 3
        logits = torch.randn(2, 4, 32, requires_grad=True)
        t = torch.randn_like(logits).log_softmax(-1)
        s = logits.log_softmax(-1)
        actual = trainer._compute_token_divergence(t, s, cfg)
        values, idx = t.topk(3, dim=-1)
        expected = (values.exp() * (values - s.gather(-1, idx))).sum(-1)
        torch.testing.assert_close(actual, expected)
        actual.mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_training_step_updates_tiny_student(self):
        import json
        from accelerate import Accelerator
        from transformers import AutoModelForMaskedLM
        from dopd.modeling import A2DQwen3Config, _register_a2d_model_classes
        _register_a2d_model_classes()
        cfg_dict = self.model_cfg.to_dict(); cfg_dict.pop('model_type', None)
        student = AutoModelForMaskedLM.from_config(A2DQwen3Config(**cfg_dict))
        accelerator = Accelerator(cpu=True)
        class Tokenizer:
            pad_token_id = 0
            mask_token_id = 31
            def convert_tokens_to_ids(self, token): return 2
            def __call__(self, texts, **kw):
                ids = [[3, 4] if text == 'prompt' else [5, 6, 7, 8] for text in texts]
                return {'length': [len(x) for x in ids]} if kw.get('return_length') else {'input_ids': torch.tensor(ids)}
        from dopd.prompting_utils import UniversalPrompting
        tok = Tokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = training_config(load_config(CONFIG), 1)
            cfg.experiment.project = tmp
            cfg.training.hindsight_top_k = 3
            cfg.training.top_k_logits = 3
            cfg.training.max_gen_length = 4
            (Path(tmp) / 'temp_data').mkdir()
            (Path(tmp) / 'temp_data/rollouts.json').write_text(json.dumps([
                dict(prompt='prompt', response='response', reward=1., hindsight_correctness=True, step_map=[0, 1, 2, 3])]))
            optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
            state = dict(accelerator=accelerator, model=student, teacher_model=self.teacher,
                hindsight_scorer_model=self.teacher, tokenizer=tok, optimizer=optimizer,
                lr_scheduler=scheduler, uni_prompting=UniversalPrompting(tok), mask_id=31,
                pad_id=0, student_total_params=1, teacher_total_params=1,
                cumulative_loss_tokens=0, wandb_enabled=False)
            before = student.lm_head.weight.detach().clone()
            with patch('torch.distributed.all_reduce'), patch.object(trainer, 'save_checkpoint'), patch.object(trainer, 'prune_epoch_checkpoints'):
                trainer.train_one_step(state, cfg)
            self.assertFalse(torch.equal(before, student.lm_head.weight))
            self.assertGreater(state['cumulative_loss_tokens'], 0)


if __name__ == '__main__':
    unittest.main()
