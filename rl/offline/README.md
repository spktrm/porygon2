# rl/offline — the replay export and what trains on it

The export is the only form on disk (`replays/README.md`); everything
derived from it is rebuilt in memory at startup. The model is the live
player model: nothing here has its own architecture.

| module | what it is | who reads it |
|---|---|---|
| `shards.py` | the service's export (`replays/shards/<format>/shard-NNN.bin`, `[uint32-LE len][EnvironmentBatch]` per replay, both perspectives): manifest check, shard listing, record iterator, byte offsets, the per-game holdout hash | `dataset.py`, `event_audit.py`, tests |
| `dataset.py` | `load_replay_store`: decodes every trajectory's terminal state ONCE (the whole event stream + `rl/environment/event_labels.py` labels) across a CPU-only spawn pool (~12 s, 8 workers, 3 GB) into `ReplayStore`, which serves padded `ReplayBatch`es; both perspectives of a game share a batch; `max_history_steps` is the load-time trailing window | `train.py` |
| `train.py` | the ONE offline trainer: `OfflineTrainer` = the player model's public path (encoder + trunk on the 55 public rows, overlaid from `--trunk-ckpt` and frozen unless `--joint`), the public critic `public_v_head` on every valid event state, and the event world model behind `--world-model` (default on; `--no-world-model` = the offline critic). Artifacts under `ckpts/offline/<format>/ckpt_NNNNNNNN` in the learner's checkpoint layout | `harness.load_search_params`, the learner's params-mode load (`merge_params` by path), `scripts/modal_train.py` |
| `config.py` | `Porygon2OfflineConfig`, the one offline config | `train.py` |
| `harness.py` | play games with plain params against a second service (`PORT=8081`), re-run the train=True heads over chunks, `search_arm` for the depth-1 rollouts | `rl/probes/*`, `search_ablation.py` |
| `search_ablation.py` | the three-arm search read (plain / search / value-blind), Wilson intervals | — |
| `event_audit.py` | corpus numbers for the event world model (kind shares, events per turn, snapshot lag) | — |
| `position_potential.py` (+ json) | the python reference of the service's position-potential fit (`service/src/server/position_potential.ts`), test-pinned | `tests/test_position_potential.py` |

Runs:

```
env/bin/python -u rl/offline/train.py --trunk-ckpt ckpts/gen9/ckpt_00373138            # critic + world model
env/bin/python -u rl/offline/train.py --trunk-ckpt ckpts/gen9/ckpt_00373138 --no-world-model
env/bin/python -m rl.offline.search_ablation --checkpoint ... --world-model ckpts/offline/...
```

`python -u`: stdout is block-buffered through tee otherwise. A script that
calls `load_replay_store` must guard its entry point (`if __name__ ==
"__main__"`): the spawned workers re-import it.

Human replays train ONLY what is listed above (CLAUDE.md, second scoped
exception): the self-play policy's losses never see a replay-derived
signal.

History: the separate offline critic architecture (margin probe, survival /
next-action / unseen / set heads, ensemble, `ckpts/offline/*-ens*`) and the
on-disk decoded layer were deleted 2026-09-16 — LESSONS "Removal ledger —
2026-09-16 offline tidy", tag `pre-offline-tidy-2026-09-16`.
