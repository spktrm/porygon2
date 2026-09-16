# replays/ — the offline export, one dataset in two forms

Everything under here is gitignored except this file and `main.py`.
The two layers are ONE dataset at two stages: the raw replay and the
service's export of it. Nothing derived is stored beside them -- a consumer
decodes the export in memory at startup (`rl/offline/dataset.py::
load_replay_store`, ~15 s across 16 processes for the whole corpus) and
chooses its own resolution there.

```
replays/
  main.py                          downloader: replay.pokemonshowdown.com -> data/
  data/<format>/<format>-<id>.json raw replay JSON (id, players, log, rating, ...)
  shards/<format>/
    manifest.json                  export_commit, num_history, feature_counts, counts
    shard-NNN.bin                  the service's export: [uint32-LE len][EnvironmentBatch]
                                   per replay, both perspectives, one EnvironmentState per
                                   committed history edge (major arg), holding that edge's
                                   effects, labelled by INFO_FEATURE__HISTORY_STEP_COUNT
                                   (the 1-based edge index; a turn boundary's slice is taken
                                   after the |turn| line); the last is the terminal state and
                                   ONLY it carries the history caches (the whole event
                                   stream); public view only (private blocks zero,
                                   all-ones mask)
```

| layer | written by | read by |
|---|---|---|
| `data/` | `replays/main.py` | the exporter; `rl/probes/battle_stats.py` (log parser) |
| `shards/*.bin` | `service/src/scripts/offline.ts` (`npm run offline -- <format>`) | `rl/offline/dataset.py::load_replay_store` (every consumer: the offline trainer, `rl/offline/event_audit.py`) |

Rules that keep the layers honest:

- `shards/manifest.json` carries the export commit and the proto feature
  counts; `rl/offline/shards.py::check_shard_manifest` refuses any other
  layout (an old export decoded silently to garbage once).
- Re-export after any proto feature change; move the old directory aside
  rather than mixing. A change to `rl/environment/event_labels.py` or to
  `process_state` needs nothing: the store decodes at startup.
- A trajectory's packed rows are addressed by ABSOLUTE index from its
  field steps (`RELEVANT_ENTITY_IDX*`); windowing the two axes
  independently breaks every gather. The store holds the whole
  trajectory; the trailing window (`max_history_steps`) goes through
  `rl/environment/utils.py::clip_history_windows_tail` and re-derives the
  labels on the window.
- The holdout split is per GAME (both perspectives together), by a hash of
  (shard path, record index) modulo `holdout_modulus`
  (`rl/offline/shards.py::is_holdout`), so every consumer agrees.
