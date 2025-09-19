<img src="figs/logo.png" alt="EAGLE" width="220" align="left"><div align="center"><h1>&nbsp;EAGLE</h1></div>

<p align="center">
| <a href="https://arxiv.org/pdf/2401.15077.pdf"><b>EAGLE</b></a> | 
<a href="https://arxiv.org/pdf/2406.16858"><b>EAGLE-2</b></a> |
<a href="https://arxiv.org/pdf/2503.01840"><b>EAGLE-3</b></a> |
<a href="https://sites.google.com/view/
eagle-llm"><b>Blog</b></a> |
</p>


<p align="center">
  <a href="">
    <img src="https://img.shields.io/badge/Version-v3.0.0-orange.svg" alt="Version">
  </a>
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License">
  </a>
  <a href="https://github.com/SafeAILab/EAGLE/issues">
    <img src="https://img.shields.io/badge/Maintained%3F-yes-green.svg" alt="Maintenance">
  </a>
  <a href="https://github.com/SafeAILab/EAGLE/pulls">
    <img src="https://img.shields.io/badge/Contributions-welcome-brightgreen.svg?style=flat" alt="Contributions welcome">
  </a>
</p>

##

<p align="center">
  <img src="./figs/eagle3r.jpg" alt="benchmark" width="790">
</p>

EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) is a new baseline for fast decoding of Large Language Models (LLMs) with provable performance maintenance. This approach involves extrapolating the second-top-layer contextual feature vectors of LLMs, enabling a significant boost in generation efficiency. 

- EAGLE is:
  - certified by the <a href="https://github.com/hemingkx/Spec-Bench/blob/main/Leaderboard.md"><b>third-party</b></a> evaluation as the **fastest** speculative method so far. 
  - achieving **2x** speedup on <a href="https://github.com/pytorch-labs/gpt-fast"><b>gpt-fast</b></a>.
  - **3x** faster than vanilla decoding (13B).
  - **2x** faster than <a href="https://lmsys.org/blog/2023-11-21-lookahead-decoding/"><b>Lookahead</b></a> (13B).
  - **1.6x** faster than <a href="https://sites.google.com/view/medusa-llm"><b>Medusa</b></a> (13B).
    - provably maintaining the consistency with vanilla decoding in the distribution of generated texts.
    - trainable (within 1-2 days) and testable on 8x RTX 3090 GPUs. So even the GPU poor can afford it.
  - combinable with other parallelled techniques such as vLLM, DeepSpeed, Mamba, FlashAttention, quantization, and hardware optimization.

EAGLE-2 uses the confidence scores from the draft model to approximate acceptance rates, dynamically adjusting the draft tree structure, which further enhances performance.

- EAGLE-2 is:
  - **4x** faster than vanilla decoding (13B).
  - **1.4x** faster than EAGLE-1 (13B).

EAGLE-3 removes the feature prediction constraint in EAGLE and simulates this process during training using training-time testing. Considering that top-layer features are limited to next-token prediction, EAGLE-3 replaces them with a fusion of low-, mid-, and high-level semantic features. 
EAGLE-3 further improves generation speed while ensuring lossless performance.

- EAGLE-3 is:
  - **5.6** faster than vanilla decoding (13B).
  - **1.8x** faster than EAGLE-1 (13B).

<p align="center">
  <img src="./figs/e3.gif" alt="demogif" width="600">
</p>

_Inference is conducted on 2x RTX 3090 GPUs at fp16 precision using the Vicuna 13B model._


[//]: # ()
[//]: # ()
[//]: # (Using EAGLE-2, the inference speed on 2 RTX 3060 GPUs can be faster than vanilla autoregressive decoding on an A100 GPU.)

## Support
EAGLE has been merged in the following mainstream LLM serving frameworks (listed in alphabetical order).

- <a href="https://rocm.docs.amd.com/en/latest/">AMD ROCm</a>
- <a href="https://angelslim.readthedocs.io/zh-cn/latest/features/speculative_decoding/eagle.html">AngelSlim</a>
- <a href="https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/feature-guide.html#eagle-speculative-decoding">AWS NeuronX Distributed Core</a>
- <a href="https://github.com/OpenBMB/CPM.cu">CPM.cu</a>
- <a href="https://github.com/intel/intel-extension-for-transformers/pull/1504">Intel® Extension for Transformers</a>
- <a href="https://github.com/intel-analytics/ipex-llm/pull/11104">Intel® LLM Library for PyTorch</a>
- <a href="https://llm.mlc.ai/docs/deploy/rest.html">MLC-LLM</a>
- <a href="https://docs.nvidia.com/nemo-framework/user-guide/latest/model-optimization/speculative/speculative.html">NVIDIA NeMo Framework</a>
- <a href="https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/eagle">NVIDIA TensorRT-LLM</a>
- <a href="https://nvidia.github.io/TensorRT-Model-Optimizer/guides/7_speculative_decoding.html">NVIDIA TensorRT Model Optimizer</a>
- <a href="https://paddlenlp.readthedocs.io/en/latest/llm/docs/predict/speculative_decoding.html">PaddleNLP</a>
- <a href="https://docs.sglang.ai/advanced_features/speculative_decoding.html">SGLang</a>
- <a href="https://github.com/sgl-project/SpecForge">SpecForge</a>
- <a href="https://github.com/vllm-project/vllm/pull/16937">vLLM</a>




## Reference
For technical details and full experimental results, please check [the paper of EAGLE](https://arxiv.org/pdf/2401.15077.pdf), [the paper of EAGLE-2](https://arxiv.org/pdf/2406.16858), and [the paper of EAGLE-3](https://arxiv.org/pdf/2503.01840).
```
@inproceedings{li2024eagle, 
  author = {Yuhui Li and Fangyun Wei and Chao Zhang and Hongyang Zhang}, 
  title = {{EAGLE}: Speculative Sampling Requires Rethinking Feature Uncertainty}, 
  booktitle = {International Conference on Machine Learning},
  year = {2024}
}
@inproceedings{li2024eagle2, 
  author = {Yuhui Li and Fangyun Wei and Chao Zhang and Hongyang Zhang}, 
  title = {{EAGLE-2}: Faster Inference of Language Models with Dynamic Draft Trees}, 
  booktitle = {Empirical Methods in Natural Language Processing},
  year = {2024}
}
@inproceedings{li2025eagle3,
      title={{EAGLE-3}: Scaling up Inference Acceleration of Large Language Models via Training-Time Test}, 
      author={Yuhui Li and Fangyun Wei and Chao Zhang and Hongyang Zhang},
      booktitle = {Annual Conference on Neural Information Processing Systems},
      year={2025},
}
```
