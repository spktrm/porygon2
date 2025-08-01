# RL Transformer Agent

A JAX/Flax implementation of a transformer-based reinforcement-learning agent designed for a structured environment with rich entity and action graphs (e.g. Pokémon-style turn-based battles).

## Folder layout

| File         | Purpose                                                                                                                                         |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.py`  | Builds a hierarchical `ConfigDict` describing the full model architecture and hyper-parameters.                                                 |
| `encoder.py` | Encodes raw environment observations (entities, edges, history) into dense latent embeddings using multiple Transformer encoder/decoder blocks. |
| `heads.py`   | Policy and value heads that map latent embeddings to action logits and scalar value estimates.                                                  |
| `model.py`   | Combines encoder and heads into a full RL network with train and inference passes plus utilities for parameter counting.                        |
| `modules.py` | Core neural-network building blocks (attention, feed-forward, SwiGLU, etc.) shared across the codebase.                                         |
| `profile.py` | Utility script that compiles the model and prints FLOPs to help with performance profiling.                                                     |
| `viz.py`     | Visualises XLA cost analysis for quick model inspection.                                                                                        |

## Profiling & FLOP counting

```bash
python profile.py   # Prints FLOP estimate
python viz.py       # Dumps XLA cost analysis
```

Both scripts compile the network once and can be adapted for TensorBoard tracing.
