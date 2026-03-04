<p align="center">
  <img src="./images/visual_interface.png" alt="CLT banner" width="800"/>
</p>

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.11-blue">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/framework-pytorch-red">
  </a>
  <a href="https://python-poetry.org/">
    <img src="https://img.shields.io/badge/packaging-poetry-cyan">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg">
  </a>
</p>

**CircuitLab** is a Mechanistic Interpretability Toolkit for jointly training [Cross-Layer Transcoders](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) (CLTs), running the auto-interp, computing the attribution-graph, and interact with a visual interface. We will soon release open-source CLTs with the library (up to 8B parameters), along with automated interpretation and code for interacting with existing open-source CLTs (e.g., Gemma 2B), enabling direct comparisons within a unified framework. We are currently adding extensions for lower-compute academic budget (e.g. low-rank finetuning of CLTs, ...). 

We believe that a major limitation in the development of CLTs, and more broadly attribution graph methods, is the significant engineering effort required to train, analyze, and iterate on them. This library aims to reduce that overhead by providing a clean, scalable, and extensible framework for academia. 

## Quick Start

### 1. Generate and cache activations

``` python
from circuitlab import ActivationsStore, clt_training_runner_config, load_model

# Load model
model = load_model("meta-llama/Llama-3.2-1B", device="cuda")

# Create config
cfg = clt_training_runner_config()

# Create activation store
store = ActivationsStore(model, cfg)

# Generate and cache activations
store.generate_and_save_activations(
    path=cfg.cached_activations_path,
    use_compression=True,  # optional
)
```

------------------------------------------------------------------------

### 2. Train the CLT

``` python
from circuitlab import CLTTrainingRunner

# Train
trainer = CLTTrainingRunner(cfg)
trainer.run()
```

------------------------------------------------------------------------

### 3. Running the AutoInterp

``` python
from circuitlab import AutoInterp, AutoInterpConfig

# Create config
cfg = AutoInterpConfig(
  model_name = model_name,
  clt_path = "path/to/checkpoint",
)

# Generate
autointerp = AutoInterp(cfg)
autointerp.run("where/to/save")
```

------------------------------------------------------------------------

### 4. Computing the Attribution Graph

``` python
from circuitlab import AttributionRunner

runner = AttributionRunner(
  model_name = model_name,
  clt_path = "path/to/checkpoint",
)
graph = runner.run(
  input_str = 'The opposite of "large" is ',
  folder = "where/to/save"
)
```

------------------------------------------------------------------------

### 5. Start the Visual-Interface

``` python
from circuitlab.frontend import main, AppConfig

cfg = AppConfig(
  graph_path = "path/to/graph", 
  autointerp_path = "path/to/autointerp"
)

main(cfg)
```

------------------------------------------------------------------------

## Features

This library currently implements L1-regularized [JumpReLU](https://arxiv.org/pdf/2407.14435) CLTs with the following design principles:

- Follows Anthropic's [training guidelines]((https://transformer-circuits.pub/2025/january-update/index.html))
- Supports feature sharding across GPUs (as well as DDP and FSDP)  
- Includes activation caching and compression/quantization of the activations  
- Adopts a structure similar to [SAE Lens](https://github.com/jbloomAus/SAELens) (code design, activation-store, etc.) and uses [Transformer Lens](https://github.com/TransformerLensOrg/TransformerLens)
- Includes a visual interface for exploring features and attribution graphs:
  - Similar in spirit (but simpler) to [Neuronpedia](https://github.com/hijohnnylin/neuronpedia)
  - Soon including attention-attribution support (as in [SparseAttention](https://arxiv.org/abs/2512.05865))

We welcome contributions to the library. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and templates. If you are interested in collaboration, you can also request access to the following [document](https://docs.google.com/document/d/1-qi6uROKHPxac0zID1EcSASDpi4Q2RIa-nDFiXJq2bw/edit?usp=sharing) with cool CLT improvement ideas. Finally, if you have any questions or want to discuss potential improvements/collaboration, write to us on the [librabry discord](https://discord.gg/UySubHGP) !

------------------------------------------------------------------------

## ⚙️ Notes
- Training happens in **multiple steps**:
  1.  **Precompute activations** (should be parallelized across indepedent jobs)
  2.  **Train the CLT model** on the cached activations (should run on a single multi-gpu node)
  3.  **Run the AutoInterp** (should be parallelized across indepedent jobs)
  4.  **Compute the Attribution-Graph** (runs on a single GPU)
  5.  **Visualize the Attribution-Graph**

-   We provide screenshot examples of training metrics in the [output](./outputs) folder and sample training scripts in [runners](./runners/)
-   Compression is optional but recommended for large-scale runs (e.g. 1B +) with 4-8x memory reduction
-   Training with bf16 is fine (autocasting with activations and weights in bf16 but gradient states in 32) but requires higher lr (around 1.5-2x bigger)
-   For Llama 1B, on a full 8 gpu H100 node, we reach an expansion factor of 42 with micro-batch size 512
-   The Visual-Interface is a simple python Dash code that is easily modifiable for your projects ! 

## Citation

<!-- ```bibtex
@misc{CircuitLab,
  title   = {CircuitLab: A Scalable Framework for Cross-Layer Transcoders Training and Attribution-Graph Visualization},
  author  = {Draye, Florent and Harasse, Abir and },
  year    = {2026},
  url     = {https://github.com/CLT-Training/CLT/}
} -->

<!-- @misc{CircuitLab2026,
  title   = {CircuitLab: A Scalable Framework for Cross-Layer Transcoders Training and Attribution-Graph Visualization},
  author  = {Florent Draye and Abir Harrasse and Vedant Palit and TY Wu and Jiarui Liu and PS Pandey and Roderick Wu and Zhijing Jin and Bernhard Sch{\"o}lkopf},
  year    = {2026},
  url     = {https://github.com/CLT-Training/CLT/}
} -->
