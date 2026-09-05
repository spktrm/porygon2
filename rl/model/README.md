# The player model

A JAX/Flax transformer over one sequence of 61 rows, one row per THING on the
board: a CLS row, 12 public entities, my 6 sheet rows, my 16 candidate move
slots, the 17 target slots, the current field triple, the recurrent field
triple, the previous action and the request info. 13.47M parameters.

## Folder layout

| File | Purpose |
| ---- | ------- |
| `constants.py` | The sequence layout — the row groups, the derived offsets, and the named slices each head reads. Every `NUM_*` is a `len()`, never a literal. |
| `config.py` | The `ConfigDict` describing the architecture. |
| `encoder.py` | Feature embedders, the entity-local pools, the recurrent history encoder, and `_assemble_sequence` — everything that turns a proto observation into the 61 rows. |
| `trunk.py` | The trunk: N unshared pre-RMSNorm blocks (self-attention + one shared SwiGLU MLP), no gates, no block masks. |
| `heads.py` | `FlatActionReadout` (a scalar per sheet row for switching, one bilinear for moves x targets, a scalar per target row for pass/default) and the categorical value head, which reads the CLS row and nothing else. |
| `player_model.py` | Trunk + readouts, sampling, and the doubles two-stage dispatch. Run it for parameter counts. |
| `history_encoder.py` | The per-slot GRU scan over the packed history cache. |
| `modules.py` | Generic primitives only — attention, SwiGLU, RMSNorm, pointer logits. Architecture lives next to its wiring. |
| `builder_model.py` | The team builder, a separate network. |
| `features.py`, `utils.py` | Feature encodings and small shared helpers. |
| `profile.py` | Compiles the model and prints a FLOP estimate. |
| `capacity.py` | Parameter-count breakdown helpers. |

## Parameter counts and profiling

```bash
env/bin/python -m rl.model.player_model   # per-module parameter counts
env/bin/python -m rl.model.profile        # FLOP estimate
```

Attention maps over a checkpoint live in `scripts/attn_probe.py`.
