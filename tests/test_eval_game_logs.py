"""The eval slate's per-game reads (rl/online/main.py eval_game_logs) score
the REAL acted rows only: the terminal row and the trailing terminal-copy
padding are out of every ratio."""

import numpy as np

from rl.environment.data import NUM_ACTION_CELLS, NUM_SWITCH_CELLS
from rl.environment.interfaces import (
    PlayerActorOutput,
    PlayerAgentOutput,
    PlayerEnvOutput,
    PlayerPolicyHeadOutput,
    PlayerTransition,
    Trajectory,
)


def test_eval_game_logs_read_real_acted_rows() -> None:
    from rl.online.main import eval_game_logs

    num_rows = 6
    done = np.zeros(num_rows, bool)
    done[3:] = True  # row 3 is the terminal row; 4-5 copy it (padding)
    action_mask = np.zeros((num_rows, NUM_ACTION_CELLS), bool)
    action_mask[:, NUM_SWITCH_CELLS] = True  # a move is always legal
    action_mask[[0, 2], 0] = True  # rows 0 and 2 offer a switch
    action_index = np.array([0, NUM_SWITCH_CELLS, NUM_SWITCH_CELLS, 0, 0, 0])
    trajectory = Trajectory(
        player_transitions=PlayerTransition(
            env_output=PlayerEnvOutput(done=done, action_mask=action_mask),
            agent_output=PlayerAgentOutput(
                actor_output=PlayerActorOutput(
                    action_head=PlayerPolicyHeadOutput(action_index=action_index),
                )
            ),
        ),
        game_length=np.array([10]),
        game_step_offset=np.array([6]),  # 4 real rows: 0..3
    )

    logs = eval_game_logs(trajectory, 2.0, "s")
    # Two decisions offered a switch (rows 0, 2), one took it; the terminal
    # row and the padding are out.
    assert logs["switch-frac-s"] == 0.5 and logs["ms-per-step-s"] == 500.0
