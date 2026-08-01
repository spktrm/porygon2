### Actor Loop Pseudocode

```python
builder_env = BuilderEnv()
player_env = PlayerEnv()

builder_model = BuilderModel()
player_model = PlayerModel()

build_traj = []

builder_env_output = builder_env.reset() # [1, B, ...]
for _ in range(6): # Always 6 for num party slots
    builder_model_output = builder_model(builder_env_output)
    builder_env_output = builder_env.step(builder_model_output.action)
    build_traj.append(BuilderTransition(builder_env_output, builder_model_output))


player_env_history = init_player_history() # [T, B, ...]
player_traj = []

for _ in range(num_player_steps):
    player_model_input = PlayerModelInput(
        player_env_output, player_env_history
    )
    player_model_output = player_model(player_model_input) # [1, B, ...]
    player_env_output, player_env_history = player_env.step(player_model_output.action)
    player_traj.append(PlayerTransition(player_env_output, player_model_output))


traj = Trajectory(
    builder=build_traj,
    player=player_traj,
    player_history=player_env_history
)
```
