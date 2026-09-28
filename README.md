# d-OPD: Future-aware on-policy distillation for block diffusion language models

d-OPD studies how to transfer the knowledge of an autoregressive teacher to a block-diffusion student. The two models predict under different contexts: an autoregressive model proceeds from left to right, while a diffusion model can use revealed tokens on both sides of a masked position. Directly matching autoregressive token distributions does not fully account for this difference.

Our approach refines the teacher's supervision using future context from the student's own rollouts. For a masked position, we consider candidate token substitutions and ask how well each candidate explains the subsequent tokens within the same block. These future-likelihood scores adjust the teacher distribution, providing a target that reflects dependencies within the student's generation process.

The method combines three components:

- **On-policy trajectories.** Distillation follows the student's generated responses and intermediate masked states, keeping supervision tied to the contexts the student encounters.
- **Future-aware teacher correction.** Candidate tokens are drawn from the teacher and student distributions. An autoregressive scorer measures their compatibility with the rollout's future tokens, and the resulting scores reweight the teacher target. Shared prefix caches reduce repeated computation across candidates.
- **Correctness-gated supervision.** The correction can be restricted to correct responses, while other responses retain the base distillation target. This limits the use of unsuccessful continuations as evidence for refining supervision.

The student learns from the corrected targets through a token-level distillation objective. The central idea is to adapt teacher supervision to the student's blockwise prediction context while retaining the teacher's autoregressive knowledge.

The core implementation is in `dopd/core.py`, with method parameters in `configs/method.yaml`. The training entrypoint is `train.py`.

For environment setup on Linux with Conda and an NVIDIA GPU:

```bash
bash setup_env.sh
```

This creates an isolated environment, installs the CUDA toolchain and dependencies, and checks the required kernels. It does not download datasets or model weights. The software stack follows the [PyTorch CUDA 12.4 installation](https://pytorch.org/get-started/previous-versions/#v260), [FlashAttention installation](https://github.com/Dao-AILab/flash-attention/tree/v2.7.4.post1#installation-and-features), and [DeepSpeed build instructions](https://www.deepspeed.ai/tutorials/advanced-install/).
