---
layout: default
title: Home
nav_order: 1
description: "LightCode Documentation"
author: "Ryan Tomich"
date: 2025-01-23
---

# LightCode: Compiling LLM Inference for Photonic-Electronic Systems
Ryan Tomich, Zhizhen Zhong, Dirk Englund
Research Laboratory of Electronics, Massachusetts Institute of Technology

<img src="Architecture.png" alt="Architecture" width="500" style="display: block; margin: auto;" />



The relentless demand for low-latency, energy-efficient inference in large language models (LLMs) has spurred interest in heterogeneous computing architectures. Although Graphics Processing Units (GPUs) currently dominate LLM deployment, their energy inefficiency and poor compatibility with emerging domain-specific accelerators like the Photonic Tensor Unit (PTU) limit their effectiveness. PTUs offer high-throughput, low-power execution of linear op- erations, but cannot natively support nonlinear and control-intensive components of LLM inference. This asymmetry motivates a hybrid compilation strategy that leverages the complementary strengths of both photonic and electronic devices. To address this challenge, we present LightCode, a compiler optimization framework and hardware-aware simulator for mapping LLM inference workloads across hybrid photonic–electronic systems. LightCode introduces a novel intermediate representation—the Stacked Graph—that encodes multiple hardware-specific realizations of each tensor operation. It formulates hardware assignment as a constrained subgraph selection problem to optimize either execution time or energy under parametric hardware cost models. We evaluate LightCode on the prefill stage of GPT-2 and Llama-7B inference using a simulated hybrid photonic–electronic architecture, measuring energy consumption, execution time, and hardware assignment under varying sequence lengths and photonic multiplexing factors. Under the assumptions of this study, we find that (i) photonic hardware can substantially reduce energy consumption; (ii) latency depends on both the degree of photonic multiplexing and the specific hardware assignment strategy; and (iii) optimization targets significantly influence the resulting hardware mappings, underscoring the need for target-aware compilation in heterogeneous systems. LightCode provides a modular, extensible foundation for compiling LLMs to next-generation architectures, including analog and photonic accelerators.

Source code available at [GitHub repository](https://github.com/RyanTomich/LightCode).

## Documentation
- [How-To Guides](how_to.md)
- [Visualizations](model_visualizations.md)
