# CLT

**CLT** is a Python library for training Cross-Layer Transcoders (CLTs) at scale.

We believe that a major limitation in the development of CLTs, and more broadly attribution graph methods, is the significant engineering effort required to train, analyze, and iterate on them. This library aims to reduce that overhead by providing a clean, scalable, and extensible framework.

## Features

This library currently implements L1-regularized CLTs with the following design principles:

- Follows Anthropic-inspired training guidelines  
- Supports feature sharding across GPUs (as well as DDP and FSDP)  
- Includes activation caching and compression/quantization of the activations  
- Adopts a structure similar to [SAE Lens](https://github.com/jbloomAus/SAELens) (activation normalization, modular design, etc.) and uses [Transformer Lens](https://github.com/TransformerLensOrg/TransformerLens)

## Compatibility and Tooling

Current CLTs trained with this library are compatible with [circuit-tracer](https://github.com/safety-research/circuit-tracer) workflows.

We also plan to release (March 2026):
- An automatic interpretability (auto-interp) pipeline  
- A visual interface for exploring features and attribution graphs  
  - Similar in spirit to [Neuronpedia](https://github.com/hijohnnylin/neuronpedia)
  - Including attention attribution support  

## Contributing

We welcome contributions to the library.  
Please refer to `contributing.md` for guidelines and templates.

## Quick Start

## Quick Start

```python
from runners.gpt2.config import clt_training_runner_config
from clt.clt_training_runner import CLTTrainingRunner

# --- Generate activations (run once, should be parallelized) ---
gen_cfg = clt_training_runner_config(generation=True)

# --- Train CLT --- (run on one node)
train_cfg = clt_training_runner_config()

trainer = CLTTrainingRunner(train_cfg)
trainer.run()
