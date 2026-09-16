# replays/ — the offline export, one dataset in three forms

Everything under here is gitignored except this file and `main.py`.
The three layers are ONE dataset at three stages of decoding; nothing
here is specific to a consumer, and a new consumer reads the deepest
layer that already has what it needs rather than adding a fourth.

```
replays/
  main.py                          downloader: replay.pokemonshowdown.com -> data/
  data/<format>/<format>-<id>.json raw replay JSON (id, players, log, rating, ...)
  shards/<format>/
    manifest.json                  export_commit, num_history, feature_counts, counts
    shard-NNN.bin                  the service's export: [uint32-LE len][EnvironmentBatch]
                                   per replay, both perspectives, one EnvironmentState per
                                   |turn| plus the terminal state; ONLY the terminal state
                                   carries the history caches (the whole event stream);
                                   public view only (private blocks zero, all-ones mask)
    decoded/
      manifest.json                export_commit it was decoded from (checked on load)
      shard-NNN-partKK.npz         the same trajectories decoded ONCE: the terminal state's
                                   unpadded event stream per trajectory (field steps, packed
                                   public / revealed / edge rows, offsets), the per-step
                                   event labels (rl/offline/event_labels.py), the outcome,
                                   the holdout flag and the record id
```

| layer | written by | read by |
|---|---|---|
| `data/` | `replays/main.py` | the exporter; `rl/offline/battle_stats.py` (log parser) |
| `shards/*.bin` | `service/src/scripts/offline.ts` (`npm run offline -- <format>`) | `rl/offline/dataset.py` (the offline critic, decodes per record with a prefetch thread) |
| `shards/decoded/` | `rl/offline/event_stream.py` (`env/bin/python rl/offline/event_stream.py`, 16 workers, ~11 s) | `rl/offline/train_world_model.py` via `EventStreamStore` (3 GB in memory, a batch is a slice) |

Rules that keep the layers honest:

- `shards/manifest.json` carries the export commit and the proto feature
  counts; `rl/offline/dataset.py::check_shard_manifest` refuses any other
  layout (an old export decoded silently to garbage once). `decoded/`
  records the export commit it came from and its store refuses a mismatch.
- Re-export (`shards/`) after any proto feature change; re-decode
  (`decoded/`) after any change to `rl/offline/event_labels.py` or to
  `process_state`. Move the old directory aside rather than mixing.
- A trajectory's packed rows are addressed by ABSOLUTE index from its
  field steps (`RELEVANT_ENTITY_IDX*`); windowing the two axes
  independently breaks every gather. `decoded/` stores the whole
  trajectory, so no rebasing is needed until the trailing clip in
  `rl/environment/utils.py::clip_history_windows_tail`.
- The holdout split is per GAME (both perspectives together), by a hash of
  (shard path, record index) modulo `holdout_modulus`; `decoded/` stores
  the flag so every consumer agrees.
