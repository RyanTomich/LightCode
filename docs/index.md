---
layout: default
title: Home
nav_order: 1
description: "LightCode Documentation"
author: "Ryan Tomich"
date: 2025-01-23
---

# LightCode: A Compiler Optimization Framework for Multi-Target LLM Compilation

LightCode is a compiler optimization framework designed to evaluate the speed and efficiency of compiling large language models (LLMs) for both photonic and classical computing architectures.

LightCode begins by leveraging TVM Relay to extract the computational graph from Hugging Face models. This graph is then transformed into a custom intermediate representation (IR) called a stacked graph, which serves as the foundation for optimization and scheduling. Depending on the desired objective—minimizing execution time or energy consumption—LightCode applies arithmetic hardware simulation.

A key feature of LightCode is its sequence-length search functionality. Given a static computational graph and hardware simulation, it identifies the sequence length at which one computing architecture becomes more efficient than another. This enables dynamic dispatch at runtime based on the user's prompt, optimizing execution on heterogeneous hardware.

For more details and source code, visit the [GitHub repository](https://github.com/RyanTomich/LightCode).


## Documentation

- [Installation Guide](installation.md)
- [Architecture](architecture.md)
- [How-To Guides](how_to.md)
- [Testing/Validation](validation.md)

# Background

# Simulation

# Limitations

**Computational Graph Operator Selection:**

In the Transformer architecture, tensor products are not performed sequentially. Instead, they are interspersed with other operations such as addition, normalization, transpose, and activation functions, which the photonic accelerator cannot execute. In contemporary LLM architectures, operations that can be directly accelerated by photonic hardware are rarely sequential. LightCode takes advantage of this assumption to accelerate the 'shortest path' graph search[^1]. For more capable hardware or model architectures where this assumption does not hold, LightCode must revert to a more exhaustive, albeit slower, graph search [quick_heuristic](https://github.com/RyanTomich/LightCode/blob/main/lightcode/graph_transformations.py#L283)


**Graph Caching:**

Modern LLMs contain repetitive subgraph structures due to sequential decoder and encoder layers. By optimizing only unique subgraphs and caching their results, optimization time can be significantly reduced

**Cost Model:**

LightCode uses an arithmetic hardware architecture simulator. Inputs are the type of operation, size of the tensor operands, and the hardware core average clock speed. If physical hardware is available, experiments can be run to derive a piecewise linear modelfor number of operations to time. Arithmetic simulation was chosen because cycle-accurate simulators trade simulation time for accuracy. Namely, the gem5 simulator takes over 600 minutes to simulate a transformer [1].

TVM's cost model was not suitable for LightCode because it prioritizes ranking optimization parameters using a learned model, rather than explicitly modeling physical execution time [4].

Photonic accelerators realize improvements by decreasing compute costs at the expense of data movement. Metrics like arithmetic intensity (ARI), or the ratio of computation to memory access, can also be considered [5]. The stepwise nature of photonic and GPU performance introduced by multiplexing, core architecture, memory, and tiling means that the FLOPs/bits would need to be calculated for each tensor shape in a hardware-aware fashion. Probabilistic modeling of cost with consideration of hardware interrupts, cache misses, and general non-determinism could also lead to improvements over many inference requests.


** Decoupling Selection and Scheduling for Improved Concurrency: **

The optimization [pipeline] (architecture.md) separates hardware selection from scheduling. This separation is suboptimal as it does not account for hardware concurrency when making selection decisions. Although the PHU may execute an operation faster, if it is occupied while the GPU is idle, scheduling the task on the GPU could reduce overall makespan. This is not currently modeled. One could consider combining the selection and scheduling into one stage to consider both simultaneously.

** Exploring Multi-Hardware Data and Task Parallelism: **
The LightCode hardware simulator assumes that each ‘operation’ as defined by Relay IR is to be done on one type of hardware. This enables task parallelism, but limits data parallelism. Data parallelism must be ‘hard coded’ into the graph expansion step of the optimization pipeline, which LightCode does for photonic matmul. LightCode assumes each Relay IR operation executes on a single hardware type, enabling task parallelism but limiting data parallelism across heterogeneous hardware. Due to the Transformer structure, we often have articulation nodes where no other operation can be done concurrently. In these situations, a performance speedup could be realized by splitting that operation between Photonics and GPU or Photonics and CPU to eliminate processor idle time.

# Future Work

## TVM integration:

**Operator Fusion:**

LightCode bypasses TVM at the Relay stage since it is the final hardware-agnostic layer. The transition from Relay IR to Tensor IR involves memory management and operator fusion, which would necessitate TVM support for photonics.
LightCode leaves all operations unfused to allow unimpeded access to the underlying tensor products. Future work could extend the optimization search space by incorporating operator fusion techniques, allowing for more efficient execution. The selection process could compare these more complex graphs where operations are not 1:1. For example, comparing `photonic_dense()` followed by an electronic `add()` operation to a `dense_add()`fused operation. This new search space could utilize an adapted version of the stacked graph IR to model the space.

**TVM Relax:**
TVM with Relay [7] [6] was limited in that it was designed for pre-autoregressive ML models, meaning it did not support the dynamicism central to modern LLM’s. The TVM community has started development on TVM Unity [3] to address these pitfalls.

## PyTorch integration
PyTorch 2.0 introduces improved support for dynamic computation graphs and compilation, which could enable LightCode’s optimizations to be realized given photonic hardware.
-	[torch.compile ](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)  JIT compiles PyTorch code into optimized kernels
-	[TorchDynamo]( https://pytorch.org/docs/stable/torch.compiler_dynamo_deepdive.html)- Graph Tracing part of torch.compile using the CPython Frame Evaluation API
-	TorchInductor -  Code generator for accelerator backends (OpenAI Triton)
-	[Custom Backends](https://pytorch.org/docs/stable/torch.compiler_custom_backends.html) - Create a backend function that is callable from TorchDynamo.

PyTorch Uses a JIT compiler for optimization and appears to have support for dynamic dispatch[^2] with PrivateUse1 - custom PyTorch backend dispatch key
-	[Multi-device integration ](https://pytorch.org/blog/pt-multidevice-integr ation/)
-	[New Backend Integration](https://pytorch.org/tutorials/advanced/privateuseone.html)


# References
[1]	Åleskog, C. et al. 2024. A Comparative Study on Simulation Frameworks for AI Accelerator Evaluation. 2024 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW) (May 2024), 321–328.

[2]	Ansel, J. et al. 2024. PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation. Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (La Jolla CA USA, Apr. 2024), 929–947.

[3]	Apache TVM Unity: a vision for the ML software & hardware ecosystem in 2022: 2021. https://tvm.apache.org/2021/12/15/tvm-unity. Accessed: 2025-01-29.

[4]	Chen, T. et al. 2018. TVM: An Automated End-to-End Optimizing Compiler for Deep Learning. (2018).

[5]	Kim, H. et al. 2024. Exploiting Intel Advanced Matrix Extensions (AMX) for Large Language Model Inference. IEEE Computer Architecture Letters. 23, 1 (Jan. 2024), 117–120. DOI:https://doi.org/10.1109/LCA.2024.3397747.

[6]	Roesch, J. et al. 2019. Relay: A High-Level Compiler for Deep Learning. arXiv.

[7]	Roesch, J. et al. 2018. Relay: A New IR for Machine Learning Frameworks. Proceedings of the 2nd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages (Jun. 2018), 58–68.



[^1] Parallels can be drawn to [Dijkstra's algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm) and the [group Steiner tree problem](https://www.cs.jhu.edu/~mdinitz/classes/ApproxAlgorithms/Spring2019/Lectures/lecture13.pdf) with stacks beign the gorups. with the main difference being that the hypergraph is directed.

[^2] [Dynamic Dispatching] (https://en.wikipedia.org/wiki/Dynamic_dispatch), deciding which function to run depending on runtime information, is closely related to  [Dynamic Linking]( https://en.wikipedia.org/wiki/Dynamic_linker), which decides which function to bring from memory at runtime. It is a subject in [Polymorphism](https://en.wikipedia.org/wiki/Polymorphism_(computer_science).
