# Third-party notices

This release was extracted from our modified OPDLM training tree. The original OPDLM code is copyright 2026 DIVE Lab, Texas A&M University, under the MIT license reproduced in `LICENSE`.

Derived infrastructure includes the trainer/checkpoint utilities, prompting and mask-token helpers, learning-rate schedules, the Qwen3 diffusion adapter, and the bundled block-diffusion inference backend (`dopd/jetengine`). The latter retains the JetEngine/SDAR implementation names to identify their technical origin. Our future-aware correction and d-OPD release changes are layered on this infrastructure; these files are not represented as entirely original work.

The release depends on PyTorch, Hugging Face Transformers/Accelerate, DeepSpeed, Triton, FlashAttention and other packages listed in `requirements.txt`; those distributions retain their own licenses. No third-party model weights or datasets are included.
