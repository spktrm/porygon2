# Porygon2

<p align="center">
    <img src="porygon2.png" alt="porygon2" width="300"/>
</p>

Successor from [meloetta](https://github.com/spktrm/meloetta/), hence porygon **2**

Porygon2 provides:

-   A [websocket server](https://github.com/spktrm/porygon2/tree/main/service) written in Node.js wrapped around the `sim` and `client` packages from [pkmn](https://github.com/pkmn).
-   A reinforcement learning framework for interacting with this server. Currently supports [Magnetic Mirror Descent](https://github.com/nathanlct/IIG-RL-Benchmark/blob/main/algorithms/mmd/mmd.py) in an asynchronous [IMPALA](https://github.com/google-deepmind/dm-haiku/tree/main/examples/impala) style.

## Overview

Porygon2 is a platform that simulates Pokémon battles and provides an environment for training reinforcement learning agents. It leverages the `pkmn` library for accurate game mechanics and offers a server-client architecture to facilitate interactions between agents and the simulation environment.

## Installation
TODO

3.  **Activate the Python Virtual Environment**

After running the script, your Python virtual environment is activated. If you open a new terminal session, reactivate it using:

```bash
source venv/bin/activate
```

## Training

`./sh start.sh`

### Client Setup

`./sh eval.sh`

## Evaluation

Porygon2 includes advanced evaluation features to help understand agent performance:

### Controlled Evaluation

To quantify whether your agent's improvement comes from better team building or better battle decisions, use **Controlled Evaluation**. This feature runs matches where both agents use the same fixed team, isolating playing skill from team building.

See [docs/CONTROLLED_EVALUATION.md](docs/CONTROLLED_EVALUATION.md) for detailed documentation.

**Quick Start:**
```python
# In your config
learner_config = Porygon2LearnerConfig(
    num_controlled_eval_actors=1,  # Enable controlled evaluation
)
```

This will track two metrics in Weights & Biases:
- `league_main_v_{checkpoint}_winrate` - Standard win rate (team building + playing)
- `league_main_v_{checkpoint}_controlled_winrate` - Controlled win rate (playing only)

## Scripts

The `scripts/` directory contains helper scripts for various tasks:

-   `compile_protos.sh`: Compiles protocol buffer definitions.
-   `generate_requirements.sh`: Generates `requirements.txt` files.
-   `lint.sh`: Runs code linters to ensure code quality.
-   `make_data.sh`: Generates necessary data for the project.
