---
layout: default
title: "Model Visualizations"
nav_order: 7
---

The motovation of this work comes from 2 sources. First is that computation time of LLM inference is dominated by the tensor products. Second, photonics can implement dot products efficiently if they had the stack that other computing platforms had.

<img src="image-4.png" alt="alt text" width="400" style="display: block; margin: auto;" />

<img src="image-13.png" alt="alt text" width="400" style="display: block; margin: auto;" />


The optimization pipeline for the computation graph.
<img src="image-1.png" alt="alt text" width="400" style="display: block; margin: auto;" />

One part of the optimization pipeline involves scheduling the computation graph with each nodes simulated time and assigned hardware. This can be visualized with time on the x axis and hardware core on the y.
<img src="image-9.png" alt="alt text" width="800" style="display: block; margin: auto;" />


# Every Computation in GPT2

<img src="image-11.png" alt="alt text" width="500" style="display: block; margin: auto;" />

# Structured GP2

<img src="image-12.png" alt="alt text" width="500" style="display: block; margin: auto;" />

# GP2 Split by articulation nodes

<img src="image-10.png" alt="alt text" width="500" style="display: block; margin: auto;" />

## Keeping only the Unique Subgraphs
<img src="image-5.png" alt="alt text" width="500" style="display: block; margin: auto;" />

# Every Computation in Llama-7b

<img src="image-7.png" alt="alt text" width="500" style="display: block; margin: auto;" />

# Structured Llama-7b
<img src="image-6.png" alt="alt text" width="500" style="display: block; margin: auto;" />
