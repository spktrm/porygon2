# Interactive evaluation

Run from the repository root with the Showdown login and client settings in `.env`:

```sh
sh eval.sh --search mcts --checkpoint ./ckpts/gen9/ckpt_01861967 --device gpu
```

This starts the existing challenge client and inference server in tmux session
`eval`. MCTS uses real legal actions at the root and generated latent actions at
imagined nodes. Defaults are depth 2, 64 simulations and four cached chance
samples per decision edge. It models the learned opponent distribution; it is
not a counterfactual regret solver.

Use `--search-depth 1`, `--simulations 128`, `--chance-samples 8`, or
`--temperature 1` to configure the experiment. The simulation budget must cover
all 16 root action slots. GPU inference is useful for search when the learner is
stopped; omit `--device` to use the configured CPU actor device. The default
interactive temperature remains 0.5.

`sh eval.sh` plays the plain policy from the latest checkpoint. `--search
expectimax` selects the existing search. `env/bin/python -m inference.server
--help` lists the server arguments without launching the challenge client.
