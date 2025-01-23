---
layout: default
title: Home
nav_order: 1
description: "LightCode Documentation"
author: "Ryan Tomich"
date: 2025-01-23
---

# Lightcode

LightCode is a compiler optimization framework designed to evaluate the speed and efficiency of multi-target compilation of LLMs to photonic and classic computers together.

First, LightCode utilizes TVM Relay to extract the computational graph of HuggingFace models. Then, LightCode transforms the graph to make a custom Intermediate Representation (IR) called a stacked graph.
This stacked Graph is then evaluated, optimized, and scheduled according to 'time' or 'energy' optimization settings. It utilizes arithmetic hardware simulation to produce best-guess estimates of compute time and energy consumption without the time overhead of cycle-accurate simulations.
Additionally, LightCode has a sequence-length search functionality. Given the static computational graph and the hardware simulator, LightCode infers which dimension is the sequence length and searches for the inflection point where one hardware becomes more optimal than another. This is useful information for runtime dynamic dispatch depending on the user prompt.

[GitHub repository](https://github.com/RyanTomich/LightCode)

## Documentation

- [Installation Guide](installation.md)
- [Architecture](architecture.md)

## Where LightCode is going
