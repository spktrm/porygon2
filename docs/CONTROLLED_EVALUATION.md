# Controlled Evaluation: Separating Team Quality from Play Quality

## Overview

This feature allows you to quantify whether an agent's improvement is due to better team building or better battle piloting (playing skill).

## The Problem

In Pokemon battles, an agent's success depends on two factors:
1. **Team Quality**: How good is the team composition (Pokemon selection and movesets)?
2. **Play Quality**: How well does the agent make decisions during battle?

When comparing different checkpoints of a trained agent, it's important to separate these two factors. A newer checkpoint might win more often simply because it built a better team, not because it plays better.

## The Solution

We implement **controlled evaluation** where both agents use the same fixed team. This isolates playing skill by removing the team building variable.

### Metrics Tracked

1. **Standard Win Rate** (`league_main_v_{checkpoint}_winrate`)
   - Regular matches where each agent builds and uses their own team
   - Measures combined improvement (team building + playing)

2. **Controlled Win Rate** (`league_main_v_{checkpoint}_controlled_winrate`)
   - Matches where both agents use the same fixed team
   - Measures pure playing skill improvement

### How to Use

#### Enable Controlled Evaluation

In your learner config, set `num_controlled_eval_actors`:

```python
learner_config = Porygon2LearnerConfig(
    num_controlled_eval_actors=1,  # Enable controlled evaluation
    # Optionally customize the fixed team used for evaluation:
    controlled_eval_fixed_species=(0, 1, 2, 3, 4, 5),  # Pokemon species indices
    controlled_eval_fixed_sets=(0, 0, 0, 0, 0, 0),     # Moveset indices
)
```

#### Customize the Fixed Team

You can customize the fixed team directly in the config (recommended):

```python
learner_config = Porygon2LearnerConfig(
    num_controlled_eval_actors=1,
    controlled_eval_fixed_species=(10, 25, 50, 75, 100, 125),  # Your team
    controlled_eval_fixed_sets=(1, 2, 0, 1, 0, 2),
)
```

#### Interpret Results

Compare the metrics in Weights & Biases:

- If **standard win rate** increases but **controlled win rate** stays flat:
  → Improvement is primarily from better team building

- If **controlled win rate** increases significantly:
  → Improvement is from better battle decisions

- If both increase:
  → Agent is improving at both team building and playing

## Example Analysis

```
Checkpoint 1000 vs Checkpoint 2000:
- Standard Win Rate: 65% → Agent 2000 wins 65% of matches
- Controlled Win Rate: 52% → With same teams, Agent 2000 barely wins

Interpretation: Most of the improvement came from better team building,
not better playing. The agent learned to build better teams but didn't
significantly improve its battle decisions.
```

## Implementation Details

### League Tracking

The `League` class now maintains two sets of statistics:
- `wins`, `draws`, `losses`, `games` - Standard matches
- `controlled_wins`, `controlled_draws`, `controlled_losses`, `controlled_games` - Controlled matches

### Actor Changes

A new method `unroll_with_fixed_team()` allows actors to play with a predetermined team rather than building one dynamically.

### Evaluation Flow

1. Controlled evaluation actor runs periodically (every 30 seconds by default)
2. It tests the main player against all historical checkpoints
3. Both players use the identical fixed team
4. Results are logged to wandb and stored in the league

## Configuration

Set `num_controlled_eval_actors` in `Porygon2LearnerConfig`:
- `0` (default): Disabled
- `1+`: Number of parallel controlled evaluation threads

Note: Controlled evaluation is more expensive than standard evaluation as it runs additional matches. Start with 1 actor.
