---
layout: default
title: "Paper Collection"
nav_order: 8
---


This is a rougly catagorizes selection of papers that informed ideas for much of this work. I tried to cite some of them throught the text, but here they all are together.

Attention is all you need [^49]
Transformers[^51]
H100 [^35]
DAC-conversions [^5]
ADC-surveys [^33]

Surveys
-	Efficient LLM serving [^32]
-	DL Compilers [^24]
-	Faster and Lighter LLMs [^6]
-	Accelerating Large Scale Generative AI [^25]
-	A survey on hardware accelerators: [^37]

Optics
-	Physics [^31]
-	DL Photonics [^47]
-	QAMNet:  Optical Nural Nets [^4]
-	Photonic Accelerators? [^30]
-	Photonic Matrix Multiplication [^65]
-	ML Photonics on chip [^12]
-	Lightning[^64]

Compilers
-	Sample free Vortex[^66]
-	Triton[^48]
-	TensorIR [^11]
-	SDSLc: multi-target [^40]
-	Roller: Tensor Compilers [^67]
-	Pytorch2 [^3]
-	Envisioning a Photonic Compiler [^27]
-	Kernals and dispatching [^57]
-	OnePerc: Photonic quantum compiler [^62]
-	oneDNN Graph Compiler [^23]
-	Photonic General  Matrix Multiply (GeMM) Compiler [^15]
-	MonoNN [^69]
-	Glow: Graph Lowering [^44]
-	Gemmini: accelerator generator [^13]
-	FlashFlex: heterogony [^53]
-	Exocompilation [^16]
-	AlpaServe: Distrabuted model serving [^26]
-	Multi-Target Compiler Architecture [^14]
-	Scheduling
    - Slice-Level [^8]
    - Workload-Aware [^17]
    - Megatron-LM Clusters [^34]
    - Alpa [^63]
-	Dynamic
    - ByteTransformer [^60]
    - DISC : A Dynamic Shape Compiler [^68]
    - Axon [^9]
    - Arlo [^46]
-	TVM[^7]
    - Relay [^43] [^42]
    - Relax [^22]
    - CIM-MLC[^38]

Simulation/benchmarking
-	VIDUR[^1]
-	GEM-5 [^28]
-	Gem5Pred [^54]
-	GPGPU and Accelerator[^39]
-	Performance Interfaces [^29]
-	LLMem: Estimating GPU Memory [^21]
-	LLMCompass [^61]
-	LightSpeed [^50]
-	In-Datacenter TPU [^18]
-	Edge TPU Accelerators [^41]
-	BOOM: performance prediction [^45]
-	Simulation Frameworks Study [^2]
-	Cost models
    - TPU and XLA [^19]
    - AMX [^20]

Quantization
-	ZeroQuant [^55], ZeroQuant-HERO [^56]
-	SmoothQuant [^52]
-	Q8BERT [^59]
-	LLM.int8() [^10]
-	Inference Optimization [^36]
-	8-bit Transformer [^58]

# References

[^1]:	Agrawal, A. et al. 2024. Vidur: A Large-Scale Simulation Framework For LLM Inference. arXiv.

[^2]:	Åleskog, C. et al. 2024. A Comparative Study on Simulation Frameworks for AI Accelerator Evaluation. 2024 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW) (May 2024), 321–328.

[^3]:	Ansel, J. et al. 2024. PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation. Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (La Jolla CA USA, Apr. 2024), 929–947.

[^4]:	Bacvanski, M.G. et al. 2024. QAMNet: Fast and Efficient Optical QAM Neural Networks. arXiv.

[^5]:	Caragiulo, P. 2024. pietro-caragiulo/survey-DAC.

[^6]:	Chavan, A. et al. 2024. Faster and Lighter LLMs: A Survey on Current Challenges and Way Forward. Proceedings of the Thirty-ThirdInternational Joint Conference on Artificial Intelligence (Jeju, South Korea, Aug. 2024), 7980–7988.

[^7]:	Chen, T. et al. TVM: An Automated End-to-End Optimizing Compiler for Deep Learning.

[^8]:	Cheng, K. et al. 2024. Slice-Level Scheduling for High Throughput and Load Balanced LLM Serving. arXiv.

[^9]:	Collins, A. and Grover, V. 2022. Axon: A Language for Dynamic Shapes in Deep Learning Graphs. arXiv.

[^10]:	Dettmers, T. et al. 2022. LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. arXiv.

[^11]:	Feng, S. et al. 2022. TensorIR: An Abstraction for Automatic Tensorized Program Optimization. arXiv.

[^12]:	Fu, T. et al. 2023. Photonic machine learning with on-chip diffractive optics. Nature Communications. 14, 1 (Jan. 2023), 70. DOI:https://doi.org/10.1038/s41467-022-35772-7.

[^13]:	Genc, H. et al. 2021. Gemmini: Enabling Systematic Deep-Learning Architecture Evaluation via Full-Stack Integration. 2021 58th ACM/IEEE Design Automation Conference (DAC) (Dec. 2021), 769–774.

[^14]:	Gökçay, E. 2022. A New Multi-Target Compiler Architecture for Edge-Devices and Cloud Management. Gazi University Journal of Science. 35, 2 (Jun. 2022), 464–483. DOI:https://doi.org/10.35378/gujs.803726.

[^15]:	Guo, Z. et al. 2022. Multi-Level Encoding and Decoding in a Scalable Photonic Tensor Processor With a Photonic General Matrix Multiply (GeMM) Compiler. IEEE Journal of Selected Topics in Quantum Electronics. 28, 6: High Density Integr. Multipurpose Photon. Circ. (Nov. 2022), 1–14. DOI:https://doi.org/10.1109/JSTQE.2022.3196884.

[^16]:	Ikarashi, Y. et al. 2022. Exocompilation for productive programming of hardware accelerators. Proceedings of the 43rd ACM SIGPLAN International Conference on Programming Language Design and Implementation (San Diego CA USA, Jun. 2022), 703–718.

[^17]:	Jain, K. et al. 2024. Intelligent Router for LLM Workloads: Improving Performance Through Workload-Aware Scheduling. arXiv.

[^18]:	Jouppi, N.P. et al. 2017. In-Datacenter Performance Analysis of a Tensor Processing Unit. Proceedings of the 44th Annual International Symposium on Computer Architecture (Toronto ON Canada, Jun. 2017), 1–12.

[^19]:	Kaufman, S.J. et al. Learned TPU Cost Model for XLA Tensor Programs.

[^20]:	Kim, H. et al. 2024. Exploiting Intel Advanced Matrix Extensions (AMX) for Large Language Model Inference. IEEE Computer Architecture Letters. 23, 1 (Jan. 2024), 117–120. DOI:https://doi.org/10.1109/LCA.2024.3397747.

[^21]:	Kim, T. et al. 2024. LLMem: Estimating GPU Memory Usage for Fine-Tuning Pre-Trained LLMs. Proceedings of the Thirty-ThirdInternational Joint Conference on Artificial Intelligence (Jeju, South Korea, Aug. 2024), 6324–6332.

[^22]:	Lai, R. et al. 2023. Relax: Composable Abstractions for End-to-End Dynamic Machine Learning. arXiv.

[^23]:	Li, J. et al. 2023. oneDNN Graph Compiler: A Hybrid Approach for High-Performance Deep Learning Compilation. arXiv.

[^24]:	Li, M. et al. 2021. The Deep Learning Compiler: A Comprehensive Survey. IEEE Transactions on Parallel and Distributed Systems. 32, 3 (Mar. 2021), 708–727. DOI:https://doi.org/10.1109/TPDS.2020.3030548.

[^25]:	Li, Y. 2024. Accelerating Large Scale Generative AI: A Comprehensive Study. Northeastern University.

[^26]:	Li, Z. et al. 2023. AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving. arXiv.

[^27]:	Lima, T.F. de et al. 2020. Primer on silicon neuromorphic photonic processors: architecture and compiler. Nanophotonics. 9, 13 (Oct. 2020), 4055–4073. DOI:https://doi.org/10.1515/nanoph-2020-0172.

[^28]:	Lowe-Power, J. et al. 2020. The gem5 Simulator: Version 20.0+. arXiv.

[^29]:	Ma, J. et al. Performance Interfaces for Hardware Accelerators.

[^30]:	Makarenko, M. et al. 2023. Photonic optical accelerators: The future engine for the era of modern AI? APL Photonics. 8, 11 (Nov. 2023), 110902. DOI:https://doi.org/10.1063/5.0174044.

[^31]:	McMahon, P.L. 2023. The physics of optical computing. Nature Reviews Physics. 5, 12 (Dec. 2023), 717–734. DOI:https://doi.org/10.1038/s42254-023-00645-5.

[^32]:	Miao, X. et al. Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems. J. ACM. 37, 4.

[^33]:	Murmann, B. 2024. bmurmann/ADC-survey.

[^34]:	Narayanan, D. et al. 2021. Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM. arXiv.

[^35]:	NVIDIA H100 Tensor Core GPU Architecture Overview: https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper. Accessed: 2024-07-30.

[^36]:	Park, Y. et al. 2024. Inference Optimization of Foundation Models on AI Accelerators. arXiv.

[^37]:	Peccerillo, B. et al. 2022. A survey on hardware accelerators: Taxonomy, trends, challenges, and perspectives. Journal of Systems Architecture. 129, (Aug. 2022), 102561. DOI:https://doi.org/10.1016/j.sysarc.2022.102561.

[^38]:	Qu, S. et al. 2024. CIM-MLC: A Multi-level Compilation Stack for Computing-In-Memory Accelerators. Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (La Jolla CA USA, Apr. 2024), 185–200.

[^39]:	Ramadas, V. et al. Simulation Support for Fast and Accurate Large-Scale GPGPU & Accelerator Workloads.

[^40]:	Rawat, P. et al. 2015. SDSLc: a multi-target domain-specific compiler for stencil computations. Proceedings of the 5th International Workshop on Domain-Specific Languages and High-Level Frameworks for High Performance Computing (Austin Texas, Nov. 2015), 1–10.

[^41]:	Reidy, B. et al. Efficient Deployment of Transformer Models on Edge TPU Accelerators: A Real System Evaluation.

[^42]:	Roesch, J. et al. 2019. Relay: A High-Level Compiler for Deep Learning. arXiv.

[^43]:	Roesch, J. et al. 2018. Relay: A New IR for Machine Learning Frameworks. Proceedings of the 2nd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages (Jun. 2018), 58–68.

[^44]:	Rotem, N. et al. 2019. Glow: Graph Lowering Compiler Techniques for Neural Networks. arXiv.

[^45]:	Su, Q. et al. 2024. BOOM: Use your Desktop to Accurately Predict the Performance of Large Deep Neural Networks. Proceedings of the 2024 International Conference on Parallel Architectures and Compilation Techniques (New York, NY, USA, Oct. 2024), 284–296.

[^46]:	Tan, X. et al. 2024. Arlo: Serving Transformer-based Language Models with Dynamic Input Lengths. Proceedings of the 53rd International Conference on Parallel Processing (Gotland Sweden, Aug. 2024), 367–376.

[^47]:	The Future of Deep Learning Is Photonic - IEEE Spectrum: https://spectrum.ieee.org/the-future-of-deep-learning-is-photonic. Accessed: 2024-01-14.

[^48]:	Tillet, P. et al. 2019. Triton: an intermediate language and compiler for tiled neural network computations. Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages (Phoenix AZ USA, Jun. 2019), 10–19.

[^49]:	Vaswani, A. et al. 2023. Attention Is All You Need. arXiv.

[^50]:	Williams, C. LightSpeed: A Framework to Profile and Evaluate Inference Accelerators at Scale.

[^51]:	Wolf, T. et al. 2020. Transformers: State-of-the-Art Natural Language Processing. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations (Online, Oct. 2020), 38–45.

[^52]:	Xiao, G. et al. 2024. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. arXiv.

[^53]:	Yan, R. et al. 2024. FlashFlex: Accommodating Large Language Model Training over Heterogeneous Environment. arXiv.

[^54]:	Yan, T. et al. 2023. Gem5Pred: Predictive Approaches For Gem5 Simulation Time. arXiv.

[^55]:	Yao, Z. et al. ZeroQuant: Efﬁcient and Affordable Post-Training Quantization for Large-Scale Transformers.

[^56]:	Yao, Z. et al. 2023. ZeroQuant-HERO: Hardware-Enhanced Robust Optimized Post-Training Quantization Framework for W8A8 Transformers. arXiv.

[^57]:	Yu, F. et al. 2024. Optimizing Dynamic-Shape Neural Networks on Accelerators via On-the-Fly Micro-Kernel Polymerization. Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (La Jolla CA USA, Apr. 2024), 797–812.

[^58]:	Yu, J. et al. 2024. 8-bit Transformer Inference and Fine-tuning for Edge Accelerators. Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3 (New York, NY, USA, Apr. 2024), 5–21.

[^59]:	Zafrir, O. et al. 2019. Q8BERT: Quantized 8Bit BERT. 2019 Fifth Workshop on Energy Efficient Machine Learning and Cognitive Computing - NeurIPS Edition (EMC2-NIPS) (Vancouver, BC, Canada, Dec. 2019), 36–39.

[^60]:	Zhai, Y. et al. 2023. ByteTransformer: A High-Performance Transformer Boosted for Variable-Length Inputs. arXiv.

[^61]:	Zhang, H. et al. LLMCompass: Enabling Efficient Hardware Design for Large Language Model Inference.

[^62]:	Zhang, H. et al. 2024. OnePerc: A Randomness-aware Compiler for Photonic Quantum Computing. Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3 (La Jolla CA USA, Apr. 2024), 738–754.

[^63]:	Zheng, L. et al. Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning.

[^64]:	Zhong, Z. et al. 2023. Lightning: A Reconfigurable Photonic-Electronic SmartNIC for Fast and Energy-Efficient Inference. Proceedings of the ACM SIGCOMM 2023 Conference (New York NY USA, Sep. 2023), 452–472.

[^65]:	Zhou, H. et al. 2022. Photonic matrix multiplication lights up photonic accelerator and beyond. Light: Science & Applications. 11, 1 (Feb. 2022), 30. DOI:https://doi.org/10.1038/s41377-022-00717-8.

[^66]:	Zhou, Y. et al. 2024. Vortex: Efficient Sample-Free Dynamic Tensor Program Optimization via Hardware-aware Strategy Space Hierarchization. arXiv.

[^67]:	Zhu, H. Roller: Fast and Efficient Tensor Compilation for Deep Learning.

[^68]:	Zhu, K. et al. 2021. DISC: A Dynamic Shape Compiler for Machine Learning Workloads. arXiv.

[^69]:	Zhuang, D. and Xia, H. MonoNN: Enabling a New Monolithic Optimization Space for Neural Network Inference Tasks on Modern GPU-Centric Architectures.
