---
layout: default
title: "Installation"
nav_order: 2
---

# Installation
Navigate to the top level of the package `LightCode/` in terminal
- run the following in the terminal
```bash
pip install -e .
```

To handel the two different phases of the simulation, there are two different enviroments.
- enviroment files can be found in `/LightCode/envs`
- For interacting with TVM
```bash
conda env create -f tvm_conda.yml
```

- for interacting with the simulator
```bash
conda env create -f schedule.yml
```
