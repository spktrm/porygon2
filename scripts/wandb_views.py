"""Build the saved wandb workspace views for this project's dashboards.

Creates/refreshes two views:
  - pokemon-rl         -> "Signal health"  (training-run diagnostics)
  - pokemon-rl-offline -> "Critic health"  (offline critic / Phi ensemble)

Panel keys mirror what rl/main.py and rl/offline/train.py log; when metrics
are added or renamed, update the sections here and re-run. Each run without
an --update-url SAVES A NEW VIEW (the API matches by internal id, not
display name); superseded copies are then PRUNED automatically — after each
save, every other view in the project with the SAME display name is
deleted. Personal workspaces ("<user>'s workspace") and any differently
named views are never touched. Pass --keep-old-views to skip pruning.

Usage:
    python scripts/wandb_views.py [--entity ENTITY]
        [--update-rl-url URL] [--update-offline-url URL]
        [--keep-old-views]

Requires `pip install wandb-workspaces` and a logged-in wandb credential.
"""

import argparse

import wandb.util

# wandb 0.28 moved generate_id out of wandb.util; wandb_workspaces'
# view-name generator still looks for it there. Shim it back before the
# wandb_workspaces import so saving a view doesn't AttributeError.
if not hasattr(wandb.util, "generate_id"):
    from wandb.sdk.lib import runid

    wandb.util.generate_id = runid.generate_id

import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws
from wandb_workspaces.workspaces import internal as ws_internal

import wandb

_LIST_VIEWS_QUERY = """
query Views($entityName: String, $name: String, $viewType: String = "project-view") {
  project(name: $name, entityName: $entityName) {
    allViews(viewType: $viewType) {
      edges { node { id name displayName } }
    }
  }
}"""

_DELETE_VIEW_MUTATION = """
mutation DeleteView($id: ID) { deleteView(input: {id: $id}) { success } }"""


def prune_stale_views(entity, project, display_name, keep_id):
    """Deletes every saved view in the project whose display name matches
    the view this script just saved, except the saved copy itself — the
    duplicates left behind by past runs that saved new views instead of
    updating in place. Matching on display name is the safety boundary:
    personal workspaces ("<user>'s workspace") and hand-made views are
    never touched."""
    api = wandb.Api()
    resp = ws_internal.execute_graphql(
        api, _LIST_VIEWS_QUERY, {"entityName": entity, "name": project}
    )
    edges = ((resp.get("project") or {}).get("allViews") or {}).get("edges", [])
    for edge in edges:
        node = edge["node"]
        if node["displayName"] != display_name or node["id"] == keep_id:
            continue
        result = ws_internal.execute_graphql(
            api, _DELETE_VIEW_MUTATION, {"id": node["id"]}
        )
        ok = result.get("deleteView", {}).get("success")
        print(f"{project}: pruned stale '{display_name}' view {node['id']} (ok={ok})")


def lp(title, y, x=None, regex=None, smooth=0.9, log_y=False, range_y=None):
    """Line plot with time-weighted EMA smoothing by default (smooth=0
    disables it — use for counters, where smoothing only misleads).
    range_y pins the y-axis, e.g. (-1, 1) for R2 panels where rare huge
    negatives otherwise blow out the scale."""
    kwargs = dict(title=title, y=y or [], log_y=log_y or None)
    if x:
        kwargs["x"] = x
    if regex:
        kwargs["metric_regex"] = regex
    if range_y is not None:
        kwargs["range_y"] = range_y
    if smooth:
        kwargs["smoothing_factor"] = smooth
        kwargs["smoothing_type"] = "exponentialTimeWeighted"
        kwargs["smoothing_show_original"] = True
    return wr.LinePlot(**{k: v for k, v in kwargs.items() if v is not None})


SH = "EvalActor-simpleheuristic"


def rl_sections():
    return [
        ws.Section(
            name="0 · Beats the heuristic?",
            is_open=True,
            panels=[
                lp(
                    "Smoothed winrate vs SimpleHeuristic",
                    [f"smoothed-wr-{SH}-{i}" for i in range(3)],
                    x="lifetime_step",
                    smooth=0,
                ),
                lp(
                    "Smoothed margin (alive-mon diff)",
                    [f"smoothed-margin-{SH}-{i}" for i in range(3)],
                    x="lifetime_step",
                    smooth=0,
                ),
                lp(
                    # Payoff (-1/0/+1), not the wr keys: runs before Aug 2026
                    # logged wr as booleans, which the wandb UI renders as
                    # NaN. Smoothed payoff reads as 2*winrate - 1.
                    "Raw payoff per actor (UI-smoothed)",
                    [f"ema-payoff-{SH}-{i}" for i in range(3)],
                    x="lifetime_step",
                    smooth=0.95,
                ),
                lp(
                    "Main-params sanity check",
                    [f"main-payoff-{SH}-{i}" for i in range(3)]
                    + [f"main-margin-{SH}-{i}" for i in range(3)],
                    x="lifetime_step",
                    smooth=0.9,
                ),
                lp(
                    "Eval games played",
                    [f"games-{SH}-{i}" for i in range(3)],
                    x="lifetime_step",
                    smooth=0,
                ),
            ],
        ),
        ws.Section(
            name="1 · Value heads",
            is_open=True,
            panels=[
                lp(
                    # Main gamma=1 head (feeds advantages) vs the
                    # multi-lambda aux CE (representation shaping only)
                    # vs the counterfactual ladder rungs (private/public).
                    "Value losses (main + aux + ladder)",
                    [
                        "player_loss_v_win",
                        "player_loss_v_aux",
                        "player_loss_v_private",
                        "player_loss_v_public",
                    ],
                ),
                lp(
                    # R2 of expectations vs v-trace targets: main head
                    # and the pooled aux-lambda rows.
                    "Value R2 (main vs aux lambdas)",
                    ["player_value_head_r2", "player_aux_value_r2"],
                    range_y=(-1, 1),
                ),
                lp(
                    # Per-lambda aux R2. The lam100 (Monte Carlo) row vs
                    # the main head is the bootstrap-bias readout: a
                    # large/growing gap = critic drifting off the data; a
                    # tiny gap during a margin plateau = transfer
                    # saturation, not value miscalibration.
                    "Aux value R2 per lambda",
                    [],
                    regex="^player_aux_r2_lam",
                    range_y=(-1, 1),
                ),
            ],
        ),
        ws.Section(
            # Stage-1 acceptance dashboard (docs/q-critic-plan.md): the
            # observer Q trains but never touches the policy, so these
            # panels are pure measurement until stage 2 flips on.
            name="1.5 · Observer Q critic",
            is_open=True,
            panels=[
                lp(
                    # Acceptance: q_r2 climbing into the V head's band.
                    # Lagging early is expected (one action's target per
                    # state vs every state for V); plateauing far below
                    # means per-action values aren't extractable from the
                    # action embeddings.
                    "Q calibration vs V head (R2)",
                    ["player_q_r2", "player_value_head_r2"],
                    range_y=(-1, 1),
                ),
                lp(
                    # THE switching readout: best legal switch E[Q] minus
                    # best legal move E[Q], joint-read with switch_ratio.
                    # Gap positive while switch_ratio collapses = ratchet
                    # (stage 2's mandate); gap negative = the critic
                    # agrees with not switching on the visited data (fix
                    # is opponents/data, not the policy update).
                    "Switch-vs-move value gap & switch rate",
                    ["player_q_switch_move_gap", "switch_ratio"],
                ),
                lp(
                    # |sum(pi*E[Q]) - V|: the two heads' state-value
                    # disagreement. Should shrink to a small stable
                    # residual; growing while both R2s look fine points
                    # at target construction, not capacity.
                    "Q-V state-value agreement",
                    ["player_q_ev_gap"],
                ),
                lp(
                    # Gap discriminator 1/3 — calibration by context.
                    # Forced switches (post-faint) stay data-rich through
                    # a switch collapse; voluntary ones starve. Forced
                    # calibrated + voluntary degraded = starvation
                    # artefact (don't trust the gap); all three tracking
                    # together = the critic means it.
                    "Q calibration by context (R2)",
                    [
                        "player_q_r2_move",
                        "player_q_r2_switch_forced",
                        "player_q_r2_switch_voluntary",
                    ],
                    range_y=(-1, 1),
                ),
                lp(
                    # Gap discriminator 2/3 — how much CE gradient switch
                    # cells actually receive (the CE trains only the
                    # taken action's cell). Voluntary frac -> 0 is the
                    # starvation mechanism in the flesh.
                    "Q training coverage by modality",
                    [
                        "player_q_switch_target_frac",
                        "player_q_voluntary_switch_target_frac",
                    ],
                ),
                lp(
                    # Gap discriminator 3/3 — head-independent: mean
                    # Retrace return after a voluntary switch vs after a
                    # move, both over states offering both modalities. If
                    # the data itself says switches lose, the negative
                    # gap is honest and the fix is opponent pressure /
                    # exploration coverage, not the improvement term.
                    "Empirical returns: voluntary switch vs move",
                    [
                        "player_q_target_voluntary_switch",
                        "player_q_target_move",
                    ],
                ),
                lp(
                    # Exploration-ladder Q intake (replaced stage 4's
                    # cross-population intake 2026-08-15). explore_frac
                    # is the realised share of Q training data from the
                    # raised-temperature actors (~their share of the
                    # actor pool); r2_explore persistently below
                    # player_q_r2 means the tempered rows are too
                    # off-policy to learn from (Retrace cutting every
                    # trace) rather than free switching counterfactuals.
                    "Exploration-ladder Q intake",
                    ["player_q_explore_frac", "player_q_r2_explore"],
                ),
                lp(
                    "Q loss & head gradient",
                    ["player_loss_q", "player_q_head_gradient_norm"],
                ),
            ],
        ),
        ws.Section(
            # Stage 2 (docs/q-critic-plan.md, enabled 2026-08-18 on the
            # 1786951032 lineage; OBSERVER since 2026-08-19 — coef zeroed
            # in favour of stage-3 Q-boosting, section 1.7. The p_q
            # diagnostics stay live as the "what would the KL have
            # pushed" readout).
            name="1.6 · Q improvement term (stage 2, observer)",
            is_open=True,
            panels=[
                lp(
                    # THE worth-it readout: p_q's switch mass vs pi's own,
                    # on states offering both a switch and a non-switch
                    # action. Converging = the term is winning its
                    # argument with the policy; pi staying flat while pq
                    # sits high = the pull isn't reaching the policy yet
                    # (check the coef ramp below).
                    "p_q vs pi switch mass (both-modality states)",
                    [
                        "player_q_improve_pq_switch_mass",
                        "player_q_improve_pi_switch_mass",
                    ],
                ),
                lp(
                    # Host-ramped runtime scalar (0 -> player_q_improve_coef
                    # over player_q_improve_ramp_steps from activation).
                    # Zero for exploiter populations by construction.
                    "Improvement-term coefficient (ramp)",
                    ["player_q_improve_coef"],
                    smooth=0,
                ),
                lp(
                    # Collapsing toward 0 = p_q sharpening onto one action
                    # (a noise-amplification risk at tau=0.1 — the abort
                    # signature to watch alongside entropy/winrate below).
                    "p_q entropy",
                    ["player_q_improve_pq_entropy"],
                ),
                lp(
                    "Q-improvement loss",
                    ["player_loss_q_improve"],
                ),
                lp(
                    # Abort criteria (docs/q-critic-plan.md): a cliff in
                    # either within ~2k steps of enabling means aux-head
                    # leak — zero player_q_improve_coef, keep the observer.
                    "Abort watch: entropy & modality entropy",
                    [
                        "player_action_normalized_entropy",
                        "player_normalized_modality_entropy",
                    ],
                ),
            ],
        ),
        ws.Section(
            # Stage 3 Q-boosting advantage (docs/q-boosting-plan.md, LIVE
            # from launch on the 2026-08-19 fresh lineage, in place of the
            # stage-2 KL): PG advantage cross-faded from v-trace to
            # retrace_g − v_exp, main population only.
            name="1.7 · Q-boosting advantage (stage 3)",
            is_open=True,
            panels=[
                lp(
                    # Host-ramped cross-fade weight (0 -> 1 over
                    # player_q_boost_ramp_steps). 1.0 = advantage fully
                    # Q-anchored; drops to 0 instantly on abort
                    # (player_q_boost_enabled=False, no recompile).
                    "Blend mix (ramp)",
                    ["player_q_boost_mix"],
                    smooth=0,
                ),
                lp(
                    # Agreement with the v-trace channel it replaces.
                    # Moderate-positive is healthy (identical would make
                    # the swap pointless, anti-correlated means one of
                    # the two estimators is broken).
                    "Boost vs v-trace advantage agreement",
                    [
                        "player_q_boost_adv_corr",
                        "player_q_boost_adv_sign_agree",
                    ],
                ),
                lp(
                    # Same-order std vs the EMA-tracked (blended) channel
                    # is the scale sanity check; mean near zero is the
                    # empirical face of Thm 3.1's unbiasedness.
                    "Boost advantage scale",
                    [
                        "player_q_boost_adv_mean",
                        "player_q_boost_adv_std",
                        "player_state_adv_std",
                    ],
                ),
                lp(
                    # Thm 3.1's precondition Var_a~pi[Q] > 0: near-zero
                    # means the policy is already ~deterministic wherever
                    # Q is evaluated, so boosting has no headroom to help
                    # regardless of critic quality.
                    "Thm 3.1 headroom: Var_a~pi[Q]",
                    ["player_q_action_var"],
                ),
                lp(
                    # The 3b gate the plan wanted BEFORE going live (this
                    # lineage went live at launch instead): the Q readout
                    # should calibrate at least as well as the V head on
                    # fresh rows. q_fresh persistently below value_fresh =
                    # the blend is anchored on the worse critic — back
                    # the mix off.
                    "Calibration r2: Q fresh/replay vs V fresh",
                    [
                        "player_q_calibration_r2_fresh",
                        "player_q_calibration_r2_replay",
                        "player_value_r2_fresh",
                    ],
                ),
                lp(
                    # Outcome + abort watch, same criteria as stage 2:
                    # entropy/winrate cliff within ~2k steps of the ramp
                    # -> flag off; switch_ratio is the number this whole
                    # saga is about.
                    "Outcome watch: switch ratio & modality entropy",
                    [
                        "switch_ratio",
                        "player_normalized_modality_entropy",
                    ],
                ),
            ],
        ),
        ws.Section(
            name="2 · Optimiser guardrails",
            is_open=True,
            panels=[
                lp(
                    "Actor KL (ceiling 0.045)",
                    [
                        "player_learner_actor_backward_kl",
                        "player_learner_actor_forward_kl",
                    ],
                ),
                lp(
                    "Replay reuse (controller)",
                    ["player_replay_realised_ratio", "player_replay_max_reuses"],
                ),
                lp(
                    # Critic calibration: |main head - MC anchor| value
                    # gap. Diagnostic only since the lambda controller's
                    # removal (2026-08-14) — a large/growing gap means
                    # bootstrap bias is leaking into targets.
                    "Bootstrap gap (critic calibration)",
                    ["player_bootstrap_gap"],
                ),
                lp(
                    # UPGO (AlphaStar-style second PG term): cut_frac is
                    # the fraction of steps whose return truncated to the
                    # bootstrap — ~0 at cold start (pessimistic critic =
                    # pure MC), rising as the critic calibrates. Loss is
                    # zeroed during plasticity recovery.
                    "UPGO (cut fraction & loss)",
                    [
                        "player_upgo_cut_frac",
                        "player_loss_upgo",
                        "player_upgo_adv_std",
                    ],
                ),
                lp(
                    # Slow BT-fit auditors, never controlled on (hundreds
                    # of games per point): worst-matchup drift and BT
                    # non-transitivity are the exploitability signature
                    # of under-regularisation.
                    "Exploitability auditors (BT-fit, slow)",
                    [
                        "league_main_winrate_min",
                        "league_main_winrate_mean",
                        "league_bt_residual",
                    ],
                ),
                lp(
                    # THE collapse watch panel — with the adaptivity
                    # controller removed (2026-08-13) modality collapse
                    # has no automated backstop, only these eyes-on axes
                    # (1330 died at modality entropy 0.08; 1328 gained
                    # strength at 0.18-0.26).
                    "Entropy axes & switch rate",
                    [
                        "player_action_normalized_entropy",
                        "player_normalized_modality_entropy",
                        "switch_ratio",
                    ],
                ),
                lp(
                    # BT rating of main vs the frozen snapshot pool.
                    # rating_valid drops to 0 when the pool is unrateable.
                    "League BT rating (auditor)",
                    ["bandit_bt_rating", "bandit_rating_valid"],
                ),
                lp(
                    "Gradient / param norm",
                    ["player_gradient_norm", "player_param_norm"],
                ),
            ],
        ),
        ws.Section(
            name="3 · League & plasticity",
            is_open=True,
            panels=[
                lp(
                    "Main vs league winrates",
                    [],
                    regex="^league_main_v_.*_winrate$",
                ),
                # The learner logs the payoff matrix through a custom
                # Vega-Lite preset registered once via
                # scripts/register_wandb_charts.py (learner._get_league_
                # winrate_heatmap): plot_table under key
                # "league_winrate_heatmap" stores its table at
                # "<key>_table". Interactive grid with proper axis
                # titles and a diverging win-rate colour scale —
                # replaces both the old matplotlib MediaBrowser image
                # panel and the later confusion-matrix-preset hijack.
                wr.CustomChart(
                    query={
                        "summaryTable": {"tableKey": "league_winrate_heatmap_table"}
                    },
                    chart_name="jtwin/league-payoff-heatmap-v10",
                    chart_fields={
                        "row": "row",
                        "row_idx": "row_idx",
                        "col": "col",
                        "col_idx": "col_idx",
                        "winrate": "winrate",
                    },
                    chart_strings={
                        "title": "league payoff table (row beats column)"
                    },
                ),
                lp(
                    "Plasticity controller",
                    [
                        "plasticity_perturbation_count",
                        "plasticity_consecutive_overdue",
                        "plasticity_recovering",
                        "plasticity_recovery_winrate",
                    ],
                    smooth=0,
                ),
                lp(
                    "Representation health",
                    [
                        "plasticity_action_emb_dormant_frac",
                        "plasticity_value_emb_dormant_frac",
                        "plasticity_action_emb_srank_frac",
                        "plasticity_value_emb_srank_frac",
                    ],
                ),
                lp(
                    "Fresh vs replayed value error",
                    [
                        "plasticity_fresh_value_err",
                        "plasticity_replay_value_err",
                        "plasticity_value_err_reuse_gap",
                    ],
                ),
            ],
        ),
        ws.Section(
            name="Losses",
            panels=[
                lp("Total player loss", ["player_loss"]),
                lp(
                    "Loss components",
                    [
                        "player_loss_pg",
                        "player_loss_upgo",
                        "player_loss_kl",
                        "player_loss_magnet_kl",
                        "player_loss_v_win",
                        "player_loss_q",
                    ],
                ),
                lp("NLL sum", ["player_nll_sum"]),
            ],
        ),
        ws.Section(
            name="Ratios & trust region",
            panels=[
                lp(
                    "Clip fractions",
                    ["player_impact_clip_frac", "player_rho_clip_frac"],
                ),
                lp("ISR ESS", ["player_isr_ess"]),
                lp(
                    "Ratios",
                    ["player_learner_actor_ratio", "player_learner_target_ratio"],
                ),
                lp(
                    "Target KLs",
                    [
                        "player_learner_target_backward_kl",
                        "player_learner_target_forward_kl",
                    ],
                ),
            ],
        ),
        ws.Section(
            name="Value & advantages",
            panels=[
                lp(
                    "Value expectation",
                    ["value_expectation_mean", "value_expectation_early_mean"],
                ),
                lp(
                    "Win returns",
                    ["player_win_returns_sum", "player_win_returns_min"],
                ),
                lp(
                    "State advantage",
                    [
                        "player_state_adv_mean",
                        "player_state_adv_std",
                        "player_win_adv_std",
                    ],
                ),
                lp(
                    # Counterfactual value ladder (2026-08-16): all
                    # (privileged, the main head) vs private (deployable) vs
                    # public (history-only) — R2 per rung.
                    "Value ladder R2",
                    [
                        "player_value_head_r2",
                        "player_value_private_r2",
                        "player_value_public_r2",
                    ],
                    range_y=(-1, 1),
                ),
                lp(
                    # |all−private| prices the opponent's hidden team;
                    # |private−public| prices private info over the public
                    # record. Signed variants read systematic bias between
                    # rungs. CAVEAT (2026-08-16): rungs are independent
                    # estimators (separate query inits/gates), so gaps
                    # include an estimator component on top of the
                    # information value — judge trends, not absolute
                    # levels, and read them alongside the ladder R2 panel.
                    "Value of information",
                    [
                        "player_value_info_gap_opp_abs",
                        "player_value_info_gap_private_abs",
                        "player_value_info_gap_opp",
                        "player_value_info_gap_private",
                    ],
                ),
            ],
        ),
        ws.Section(
            name="Behaviour & environment",
            panels=[
                lp("Move / switch ratio", ["move_ratio", "switch_ratio"]),
                lp("Early finish rate", ["early_finish_rate"]),
                lp("Reward mean", ["reward_mean"]),
                lp("History & wildcard", ["history_lengths_mean", "wildcard_turn"]),
                lp(
                    # Whole-game length off terminal chunks' done rows — the
                    # distribution to watch since the 96-request force-tie
                    # was removed (2026-08-16).
                    "Game length",
                    [
                        "game_length_requests_mean",
                        "game_length_requests_max",
                        "game_length_turns_mean",
                    ],
                ),
                lp(
                    # Chunked unrolls: valid rows per 64-row chunk (padding
                    # share), terminal-chunk fraction (~1/chunks-per-game),
                    # and history-window underrun (sustained >0 means
                    # player_history_length is too small).
                    "Chunk lengths",
                    [
                        "player_trajectory_length_mean",
                        "player_trajectory_length_min",
                        "player_trajectory_length_max",
                    ],
                ),
                lp(
                    "Chunk health",
                    [
                        "player_chunk_terminal_frac",
                        "player_chunk_history_underrun",
                    ],
                ),
                lp(
                    "Masks",
                    [
                        "player_policy_mask_sum",
                        "player_value_mask_sum",
                        "player_policy_value_mask_ratio",
                    ],
                ),
            ],
        ),
        ws.Section(
            name="Gradient norms by module",
            panels=[
                lp(
                    # Keys are f"player_{module}_gradient_norm" over the
                    # param tree's top level (+ encoder submodules matching
                    # *encoder/*decoder) — see learner.py's training_logs.
                    "Module grad norms",
                    [
                        "player_encoder_gradient_norm",
                        "player_history_encoder_gradient_norm",
                        "player_macro_head_gradient_norm",
                        "player_pi_head_gradient_norm",
                        "player_v_head_gradient_norm",
                        "player_aux_v_head_gradient_norm",
                        "player_q_head_gradient_norm",
                    ],
                    log_y=True,
                ),
            ],
        ),
        ws.Section(
            name="Throughput",
            panels=[
                lp(
                    "Frame counts",
                    ["player_frame_count", "builder_frame_count"],
                    smooth=0,
                ),
                lp(
                    # training_step (player_state.step_count) resets to 0
                    # on an exploiter hard-reset/perturb re-fork;
                    # lifetime_step never does — plotted together (against
                    # the lifetime_step x-axis) a re-fork shows as
                    # training_step visibly dropping while the x-axis
                    # itself keeps climbing.
                    "Training step (raw) vs lifetime step",
                    ["training_step", "lifetime_step"],
                    smooth=0,
                ),
            ],
        ),
        ws.Section(
            # Fed by learner.py's _log_memory_diagnostics (main-only, every
            # memory_diag_interval steps) plus the service's own 10s
            # process.memoryUsage() write — see index.ts:writeMemoryStats.
            name="Memory",
            panels=[
                lp(
                    "Process RSS (MB)",
                    ["diag_rss_mb", "diag_node_rss_mb"],
                    smooth=0,
                ),
                lp(
                    # node's own heap is the tiny GameServer coordinator
                    # thread only (Node quirk — memoryUsage() can't see
                    # another isolate's heap); worker_heap_used_mb is the
                    # actual dex/sim data, summed across all 6 workers.
                    "Node heap used (MB): coordinator vs workers",
                    ["diag_node_heap_used_mb", "diag_node_worker_heap_used_mb"],
                    smooth=0,
                ),
                lp(
                    "Thread counts",
                    ["diag_os_threads", "diag_py_threads", "diag_node_num_workers"],
                    smooth=0,
                ),
                lp(
                    "Python thread buckets",
                    [],
                    regex="^diag_py_threads_",
                    smooth=0,
                ),
                lp(
                    "Replay buffer bytes (MB)",
                    [],
                    regex="^diag_(player|builder)_replay_mb_",
                    smooth=0,
                ),
                lp(
                    "League cache",
                    ["diag_league_cache_mb", "diag_league_cache_entries"],
                    smooth=0,
                ),
            ],
        ),
    ]


def members(stem):
    return [f"{stem}_mean"] + [f"{stem}_m{k}" for k in range(4)]


def offline_sections():
    return [
        ws.Section(
            name="0 · Outcome head (Φ)",
            is_open=True,
            panels=[
                lp("Held-out loss", members("eval_loss")),
                lp(
                    "Sign accuracy (all steps vs terminal)",
                    ["eval_accuracy_mean", "eval_accuracy_last_step_mean"],
                ),
                lp("Margin MAE", members("eval_margin_mae")),
                lp(
                    "Margin std (train batch)",
                    ["margin_std_mean", "announced_margin_std_mean"],
                ),
            ],
        ),
        ws.Section(
            name="1 · Ensemble & gate",
            is_open=True,
            panels=[
                lp("Member disagreement (std)", ["eval_gate_member_std"]),
                lp("Gated |Φ| (scale 5)", ["eval_gate_abs_phi"]),
                lp("Gated sign accuracy", ["eval_gate_accuracy"]),
            ],
        ),
        ws.Section(
            name="2 · Announced head (Φ_ann)",
            is_open=True,
            panels=[
                lp("Held-out announced loss", members("eval_announced_loss")),
                lp("Announced sign accuracy", members("eval_announced_accuracy")),
                lp(
                    "Distill KL (realised ↔ announced)",
                    members("eval_announced_distill_kl"),
                ),
            ],
        ),
        ws.Section(
            name="Aux heads (held-out)",
            panels=[
                lp(
                    "Survival",
                    ["eval_survival_loss_mean", "eval_survival_loss_imminent_mean"],
                ),
                lp(
                    "Unseen-move hazard",
                    ["eval_unseen_loss_mean", "eval_unseen_loss_imminent_mean"],
                ),
                lp(
                    "Revealed-set head",
                    ["eval_set_loss_mean", "eval_set_pos_prob_mean"],
                ),
                lp(
                    "Next-action loss",
                    ["eval_action_loss_mean", "eval_action_loss_unrevealed_mean"],
                ),
                lp(
                    "Next-action accuracy",
                    [
                        "eval_action_accuracy_mean",
                        "eval_action_accuracy_unrevealed_mean",
                    ],
                ),
            ],
        ),
        ws.Section(
            name="Train-side (batch, per member)",
            panels=[
                lp("Train loss", members("loss")),
                lp("Gradient norm", members("gradient_norm"), log_y=True),
                lp("Train sign accuracy", members("accuracy")),
                lp(
                    "Train survival / unseen",
                    [
                        "survival_loss_mean",
                        "survival_loss_imminent_mean",
                        "unseen_loss_mean",
                        "unseen_loss_imminent_mean",
                    ],
                ),
                lp(
                    "Train action / set",
                    [
                        "action_loss_mean",
                        "action_loss_unrevealed_mean",
                        "set_loss_mean",
                    ],
                ),
            ],
        ),
        ws.Section(
            name="Data & target counts",
            panels=[
                lp("Valid steps / batch", ["num_valid_steps_mean"]),
                lp(
                    "Aux target counts",
                    [
                        "num_action_targets_mean",
                        "num_survival_targets_mean",
                        "num_unseen_targets_mean",
                        "num_set_positives_mean",
                    ],
                ),
            ],
        ),
    ]


def save_view(entity, project, name, sections, update_url, settings=None, force_x=None):
    # Panel-level x overrides the workspace-level x_axis setting, and the
    # save/round-trip path materialises the default "Step" (wandb's row
    # counter) onto every panel that doesn't set one — silently defeating
    # WorkspaceSettings(x_axis=...). Force the axis per panel instead.
    if force_x:
        for section in sections:
            for panel in section.panels:
                # Only LinePlot has an x field — MediaBrowser (and other
                # non-line panel types) use gallery_axis/grid_x_axis/
                # grid_y_axis instead, and pydantic's validate_assignment
                # rejects setting an attribute the model doesn't declare.
                if isinstance(panel, wr.LinePlot):
                    panel.x = force_x
    if update_url:
        workspace = ws.Workspace.from_url(update_url)
        workspace.name = name
        workspace.sections = sections
    else:
        workspace = ws.Workspace(
            entity=entity, project=project, name=name, sections=sections
        )
    if settings is not None:
        workspace.settings = settings
    workspace.save()
    print(f"{project} view: {workspace.url}")
    return workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="jtwin")
    parser.add_argument("--update-rl-url", default=None)
    parser.add_argument("--update-offline-url", default=None)
    parser.add_argument(
        "--keep-old-views",
        action="store_true",
        help="Skip deleting superseded same-name views after saving.",
    )
    args = parser.parse_args()

    rl_workspace = save_view(
        args.entity,
        "pokemon-rl",
        "Signal health",
        rl_sections(),
        args.update_rl_url,
        # Learner-step x-axis for every panel: lifetime_step (not
        # training_step = player_state.step_count) — the latter resets to
        # 0 on an exploiter hard-reset/perturb re-fork (PopulationState's
        # host_step is zeroed at fork; training_step tracks it), which
        # would draw a sawtooth/overdraw on any main_exploiter/
        # league_exploiter panel. lifetime_step is carried across re-forks
        # by construction (rl/online/learner.py's PopulationState) and is
        # already the run's own default step metric (main.py's
        # define_metric("*", step_metric="lifetime_step")) — logged on
        # every row, including the eval-actor rows that only log
        # training_step themselves (wandb fills forward the run's last
        # logged lifetime_step for those). The offline project does not
        # log this key, so it keeps the default. force_x pins it per panel
        # — the workspace-level setting alone is overridden by the "Step"
        # default materialised onto each panel on save.
        settings=ws.WorkspaceSettings(x_axis="lifetime_step"),
        force_x="lifetime_step",
    )
    offline_workspace = save_view(
        args.entity,
        "pokemon-rl-offline",
        "Critic health",
        offline_sections(),
        args.update_offline_url,
    )
    if not args.keep_old_views:
        prune_stale_views(
            args.entity, "pokemon-rl", "Signal health", rl_workspace._internal_id
        )
        prune_stale_views(
            args.entity,
            "pokemon-rl-offline",
            "Critic health",
            offline_workspace._internal_id,
        )


if __name__ == "__main__":
    main()
