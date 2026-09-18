"""Does the service potential carry outcome information the critic lacks?

The PBRS channel's advantage at launch (potential head fresh at 0, gamma 1)
is the v-trace advantage plus eta * Psi_t, Psi_t the lambda-return of the
shaping rewards Phi_{t+1} - Phi_t over the rest of the game. It helps the
policy gradient exactly when Psi_t is informative about the critic's own
error e_t = G - V(s_t) against the realised outcome G. On self-play games
recorded from a stopped run's checkpoint, per row: corr(Psi, e), the share
of the residual variance Psi explains, the best eta (cov / var) and what eta
buys at the launch value; split by the taken action so the switch / stay gap
of the advantage audit can be read beside what the channel would add to it.

    PORT=8081 MAX_WORKERS=2 MEMORY_STATS_PATH=/tmp/x.json \\
        node service/dist/server/index.js
    PS_SERVICE_URI=ws://localhost:8081 env/bin/python -m rl.probes.potential_information \\
        --checkpoint ckpts/gen9/ckpt_00303473 --games 32 --out runtime/pbrs/info.json
"""

import argparse
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

from rl.offline import harness
from rl.online.config import Porygon2LearnerConfig
from rl.online.training.targets import unit_potential
from rl.online.training.telemetry import action_axis_masks
from rl.probes.uniform_kl_screen import restored_states

logger = logging.getLogger(__name__)


def side_rows(chunks, values):
    """One game-side as flat per-row arrays, the chunk overlap row dropped
    (a chunk's final row is the next chunk's first, bootstrap-only unless it
    is the done row)."""
    phi, value, done, switch, real_choice = [], [], [], [], []
    for index, (chunk, chunk_value) in enumerate(zip(chunks, values)):
        env = chunk.player_transitions.env_output
        actor = chunk.player_transitions.agent_output.actor_output
        axis = action_axis_masks(env.action_mask, actor.action_head.action_index)
        rows = np.asarray(env.done).shape[0]
        if index + 1 < len(chunks):
            rows -= 1
        phi.append(np.asarray(unit_potential(env))[:rows])
        value.append(np.asarray(chunk_value)[:rows])
        done.append(np.asarray(env.done)[:rows].astype(bool))
        switch.append(np.asarray(axis.taken_switch)[:rows])
        real_choice.append(np.asarray(axis.has_both)[:rows])
    phi, value, done, switch, real_choice = (
        np.concatenate(part) for part in (phi, value, done, switch, real_choice)
    )
    end = int(np.flatnonzero(done)[0]) if done.any() else len(done) - 1
    return (
        phi[: end + 1],
        value[: end + 1],
        switch[: end + 1],
        real_choice[: end + 1],
        harness.outcome(chunks),
    )


def shaping_return(phi, lam):
    """Psi_t = sum_k lam^k (Phi_{t+k+1} live_{t+k+1} - Phi_{t+k}), gamma 1,
    the potential 0 past the done row (targets.compute_player_targets)."""
    live_next = np.append(phi[1:], 0.0)
    live_next[-2:] = 0.0
    reward = live_next - phi
    reward[-1] = 0.0
    psi = np.zeros_like(phi)
    carry = 0.0
    for step in range(len(phi) - 1, -1, -1):
        carry = reward[step] + lam * carry
        psi[step] = carry
    return psi


def statistics(psi, residual, eta):
    if len(psi) < 3 or np.var(psi) == 0:
        return {"rows": int(len(psi))}
    cov = float(np.cov(psi, residual)[0, 1])
    corr = float(np.corrcoef(psi, residual)[0, 1])
    base = float(np.mean(residual**2))
    return {
        "rows": int(len(psi)),
        "residual_mean": float(np.mean(residual)),
        "residual_rms": float(np.sqrt(base)),
        "psi_mean": float(np.mean(psi)),
        "psi_rms": float(np.sqrt(np.mean(psi**2))),
        "corr": corr,
        "variance_explained": corr**2,
        "eta_best": cov / float(np.var(psi)),
        # Squared error of (V + eta Psi) against G relative to V's own.
        "mse_ratio_at_eta": float(np.mean((residual - eta * psi) ** 2)) / base,
        "mse_ratio_at_eta_best": float(
            np.mean((residual - cov / np.var(psi) * psi) ** 2)
        )
        / base,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    config = Porygon2LearnerConfig()
    player_state, _ = restored_states(arguments.checkpoint, config)
    sides = harness.play_games(
        player_state.params,
        arguments.games,
        pairs=4,
        tag="potential-information",
        seed=arguments.seed,
        opponent="self",
        temperature=1.0,
        device=arguments.device,
    )
    chunks = harness.flatten(sides)
    values = []
    for prediction, _ in harness.forward(player_state.params, chunks):
        expectation = np.asarray(prediction.value_head.expectation, dtype=np.float32)
        values.extend(expectation[:, column] for column in range(expectation.shape[1]))
    if not any(
        np.asarray(unit_potential(c.player_transitions.env_output)).any()
        for c in chunks
    ):
        raise SystemExit("every recorded potential is 0: rebuild service/dist")

    collected = {key: ([], []) for key in ("all", "switch", "stay", "real_choice")}
    cursor = 0
    for side in sides:
        side_values = values[cursor : cursor + len(side)]
        cursor += len(side)
        phi, value, switch, real_choice, outcome = side_rows(side, side_values)
        psi = shaping_return(phi, config.player_lambda)
        residual = outcome - value
        # The done row carries no decision.
        keep = np.ones(len(phi), dtype=bool)
        keep[-1] = False
        for key, mask in (
            ("all", keep),
            ("switch", keep & switch & real_choice),
            ("stay", keep & ~switch & real_choice),
            ("real_choice", keep & real_choice),
        ):
            collected[key][0].append(psi[mask])
            collected[key][1].append(residual[mask])

    report = {
        "checkpoint": arguments.checkpoint,
        "games": arguments.games,
        "eta": arguments.eta,
        "lambda": config.player_lambda,
    }
    for key, (psi_parts, residual_parts) in collected.items():
        report[key] = statistics(
            np.concatenate(psi_parts), np.concatenate(residual_parts), arguments.eta
        )
    Path(arguments.out).parent.mkdir(parents=True, exist_ok=True)
    Path(arguments.out).write_text(json.dumps(report, indent=2))
    for key in ("all", "real_choice", "switch", "stay"):
        logger.info("%s: %s", key, json.dumps(report[key]))


if __name__ == "__main__":
    main()
