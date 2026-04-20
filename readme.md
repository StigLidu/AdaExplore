# AdaExplore

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxx)

> **Weihua Du, Jingming Zhuo, Yixin Dong, Andre Wang He, Weiwei Sun, Zeyu Zheng, Manupa Karunaratne, Ivan Fox, Tim Dettmers, Tianqi Chen, Yiming Yang, Sean Welleck**  
> ["AdaExplore: Failure-Driven Adaptation and Diversity-Preserving Search for Efficient Kernel Generation " (2026)](https://arxiv.org/abs/xxxx.xxxxx)

AdaExplore is a research codebase for **LLM-driven GPU kernel engineering**. It combines:

- agent self-explored skill memory for mastering kernel languages
- a multi-strategy search agent for proposing and revising kernels
- a benchmark and evaluation harness for correctness and performance
- an optional remote evaluation service for multi-GPU or cluster setups

## What It Does

Given a reference PyTorch operator or model component, AdaExplore asks an LLM to generate optimized CUDA/Triton-style implementations, evaluates the generated kernels, and uses search to keep improving them.

## TODOs

- [ ] Documentation is currently in progress, including installation and usage guides. More updates will be added soon.

## License

This repository is released under the MIT License. See `LICENSE`.
