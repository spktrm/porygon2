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
                    # The one critic's CE against the v-trace win targets.
                    "Value loss",
                    ["player_loss_v_win"],
                ),
                lp(
                    # R2 of expectations vs v-trace targets.
                    "Value R2 (main head)",
                    ["player_value_head_r2"],
                    range_y=(-1, 1),
                ),
            ],
        ),
        ws.Section(
            # What is left of the critic section after the advantage head
            # retired (2026-08-29). Every panel that read a Q readout went
            # with it; these read the one-step V LABEL and the taken-action
            # coverage, which are properties of the data, not of any head --
            # which is exactly why they outlived three critic designs.
            name="1.5 · Switch evidence",
            is_open=True,
            panels=[
                lp(
                    # How much gradient switch cells actually receive. The
                    # policy loss trains only the taken action, so
                    # voluntary frac -> 0 IS the starvation mechanism in the
                    # flesh, whatever the head looks like.
                    "Training coverage by modality",
                    [
                        "player_q_switch_target_frac",
                        "player_q_voluntary_switch_target_frac",
                    ],
                ),
                lp(
                    # The flat readout's own way to fail, and it is the same
                    # SHAPE as the dx65cpwp runaway these panels were built
                    # for: the bilinear is a two-factor product with ONE
                    # zero-init factor. query must leave 0 within ~200 steps
                    # (its gradient is a rank-1 outer product of live rows);
                    # key must leave lecun 0.0625 shortly after (its gradient
                    # is proportional to query, so it is frozen for exactly
                    # one step). Either still flat at 2k IS the stall.
                    "Action readout: drift from init",
                    [
                        "player_pointer_query_rms",
                        "player_pointer_key_rms",
                        "player_pointer_local_src_rms",
                        "player_pointer_local_tgt_rms",
                        "player_switch_head_rms",
                        "player_other_head_rms",
                    ],
                ),
                lp(
                    "Trunk & head gradient norms",
                    [
                        "player_trunk_attn_out_rms",
                        "player_trunk_mlp_out_rms",
                        "player_action_head_grad_norm",
                        "player_trunk_grad_norm",
                    ],
                ),
                lp(
                    # Head-independent: mean one-step return after a
                    # voluntary switch vs after a move, over states offering
                    # both. If the DATA says switches lose, a collapsing
                    # switch rate is honest and the fix is opponent pressure
                    # and coverage, not the update rule.
                    "Empirical returns: voluntary switch vs move",
                    [
                        "player_q_target_voluntary_switch",
                        "player_q_target_move",
                    ],
                ),
                lp(
                    "Voluntary-switch rows per batch",
                    [
                        "player_q_support_vol_switch_rows",
                        "player_q_support_forced_switch_rows",
                    ],
                ),
            ],
        ),
        ws.Section(
            # THE policy gradient since 2026-08-26: NashPG
            # (arXiv:2510.18183) — a PPO-clipped surrogate on the taken
            # action's pi/mu ratio over the batch-normalised v-trace
            # advantage, plus a differentiated forward KL magnet to the
            # periodically snapped reference and an entropy bonus inside
            # the same bracket.
            name="1.6 · NashPG policy loss",
            is_open=True,
            panels=[
                lp(
                    # The surrogate's value and how often the trust region
                    # is active. clip_frac pinned near 0 = the policy
                    # barely moves (look at lr/coef before blaming the
                    # critic); climbing toward 1 = replayed data has
                    # outrun the band (staleness / replay controller).
                    "PPO surrogate & clip occupancy",
                    ["player_loss_pg", "player_ppo_clip_frac"],
                ),
                lp(
                    # Batch advantage statistics BEFORE normalisation —
                    # the scale the unit-std surrogate advantage divides
                    # out. std collapsing toward 0 = the value function
                    # sees no return differences to steer by.
                    "v-trace advantage scale (pre-normalisation)",
                    ["player_pg_adv_mean", "player_pg_adv_std"],
                ),
                lp(
                    # The magnet cycle: KL(pi || pi_reg) sawtooths — up
                    # against the FROZEN reference, ~0 at each snap. A
                    # level climbing ACROSS snaps is a policy outrunning
                    # the snap period.
                    "Magnet: reference KL sawtooth & snaps",
                    ["player_ref_kl", "player_reg_snapped"],
                ),
                lp(
                    # Entropy bonus value (alpha-weighted since 2026-08-28)
                    # and the two normalised entropies. A cliff here now
                    # means the CONTROLLER is losing at its alpha ceiling,
                    # not that a static coef needs a bump.
                    "Entropy bonus & abort watch",
                    [
                        "player_loss_entropy",
                        "player_action_normalized_entropy",
                        "player_normalized_modality_entropy",
                    ],
                ),
                lp(
                    # The per-axis normalised entropies the floor holds
                    # (targets player_ent_target_{macro,micro} = 0.5).
                    # micro_taken staying off its floor on switch-heavy
                    # batches is the which-axis plasticity signal.
                    "Entropy floor: per-axis normalised entropies",
                    [
                        "player_entropy_macro",
                        "player_entropy_micro_taken",
                    ],
                ),
                lp(
                    # The dual temperatures (2026-08-28). Equilibrium away
                    # from both bounds = the floor holds at finite cost;
                    # pinned at alpha_max (0.5) = the ask is infeasible
                    # against the PG at this bound — the abort instrument;
                    # at alpha_min = the axis holds itself for free.
                    "Entropy floor: dual temperatures alpha",
                    [
                        "player_ent_alpha_macro",
                        "player_ent_alpha_micro",
                    ],
                    log_y=True,
                ),
                lp(
                    # The zero-avoiding term: forward KL from UNIFORM, the
                    # one force in the bracket that is not pi-prefactored and
                    # so the only one still acting on an abandoned cell.
                    # Read against prob_switch -- rising mass with this
                    # falling is the term doing its job and then relaxing.
                    "Zero-avoiding KL & switch mass",
                    ["player_loss_uniform_kl", "player_policy_prob_switch"],
                ),
                lp(
                    # Modality decomposition of the two throttles on any
                    # taken-action update: per-cell pi mass and the
                    # observer critic's |A|, as switch/move ratios.
                    # prob_ratio falling while absadv_ratio holds ~ 1 is
                    # the starvation signature — the critic still
                    # believes, the policy has stopped sampling.
                    "Starvation watch (ratios, switch/move)",
                    ["player_policy_prob_ratio"],
                    log_y=True,
                ),
                lp(
                    "Starvation factors: per-cell pi and |A|",
                    [
                        "player_policy_prob_switch",
                        "player_policy_prob_move",
                    ],
                    log_y=True,
                ),
            ],
        ),
        ws.Section(
            # Observer critic quality. The policy no longer reads the Q
            # stack (2026-08-26; its link to return is the v-trace
            # advantage), but an action-flat critic still voids the
            # matched control and the starvation discriminators above.
            name="1.7 · Critic quality (observer stack)",
            is_open=True,
            panels=[
                lp(
                    # Pre-clip grad norm per policy-head subtree, the
                    # policy pathway's own gradient scale (the Q-head pair
                    # below stayed calm through both dx65cpwp failures).
                    "Policy head: grad norm by subtree",
                    [
                        "player_policy_head_gradient_norm",
                    ],
                ),
                lp(
                    # The Q readout should calibrate at least as well as
                    # the V head on fresh rows. q_fresh persistently below
                    # value_fresh = the policy is steering off the worse
                    # critic.
                    "Calibration r2: Q fresh/replay vs V fresh",
                    [
                        "player_value_r2_fresh",
                    ],
                ),
                lp(
                    # switch_ratio is the number this whole saga is about;
                    # an entropy cliff is the abort signal (raise
                    # player_ent_coef first; back player_pg_coef off).
                    "Outcome watch: switch ratio & modality entropy",
                    [
                        "switch_ratio",
                        "player_normalized_modality_entropy",
                    ],
                ),
            ],
        ),
        ws.Section(
            # Step 1 of docs/critic-weakness-analysis.md (2026-08-23): the
            # per-row JOINT statistics that judge every later step, from
            # the completed-game outcome now carried on every chunk. NaN
            # where a batch has no rows in the slice (wandb skips them).
            name="1.75 · Critic telemetry (Step 1: labels, matched-V, support)",
            is_open=True,
            panels=[
                lp(
                    # Realised outcome of voluntary switches minus moves at
                    # matched V(s). Offline: pooled -0.147 -> matched
                    # -0.048±0.054. Per-batch n is tiny; read smoothed and
                    # with the n panel beside it.
                    "Matched-V realised gap (vol switch − move) per V bin",
                    [
                        "player_mv_bin0_gap_realised",
                        "player_mv_bin1_gap_realised",
                        "player_mv_bin2_gap_realised",
                        "player_mv_bin3_gap_realised",
                        "player_mv_bin4_gap_realised",
                        "player_mv_pooled_gap_realised",
                    ],
                    smooth=0.99,
                ),
                lp(
                    "Matched-V support: voluntary switches per V bin",
                    [
                        "player_mv_bin0_n_vol",
                        "player_mv_bin1_n_vol",
                        "player_mv_bin2_n_vol",
                        "player_mv_bin3_n_vol",
                        "player_mv_bin4_n_vol",
                    ],
                    smooth=0.99,
                ),
                lp(
                    # Selection, directly: V at the states where switches
                    # are taken vs where moves are (offline -0.04 vs +0.08).
                    "V(s) at voluntary switches vs moves",
                    ["player_mv_v_at_vol_switch", "player_mv_v_at_move"],
                    smooth=0.99,
                ),
                lp(
                    # Outcome calibration of the V head (offline 0.265 on
                    # fresh on-policy games). prev_switch vs prev_move is
                    # the post-switch pessimism read.
                    "V outcome R²: all / phase / after switch vs move; V vs one-step",
                    [
                        "player_v_outcome_r2_all",
                        "player_v_outcome_r2_early",
                        "player_v_outcome_r2_mid",
                        "player_v_outcome_r2_late",
                        "player_v_outcome_r2_prev_switch",
                        "player_v_outcome_r2_prev_move",
                        "player_v_onestep_r2",
                    ],
                    smooth=0.99,
                ),
                lp(
                    # Signed bias V − G after a switch vs after a move.
                    # Negative after switches = the pessimism a V-bootstrap
                    # label would inherit (Step 3 caveat).
                    "V outcome bias (V − G) after switch vs after move",
                    [
                        "player_v_outcome_bias_prev_voluntary",
                        "player_v_outcome_bias_prev_forced",
                        "player_v_outcome_bias_prev_move",
                    ],
                    smooth=0.99,
                ),
                lp(
                    # Storage-level (chunks holding a voluntary switch) and
                    # optimisation-level (loss share) support — the
                    # acceptance measure for the Step-2 ramp and any row
                    # weighting.
                    "Q support: loss share by modality, chunk frac, edge frac",
                    [
                        "player_q_support_chunk_vol_switch_frac",
                        "player_q_target_edge_frac",
                    ],
                ),
                lp(
                    # THE DEADLINE PANEL. Voluntary-switch rows per batch is
                    # N*pi_switch, and a starved modality becomes absorbing
                    # once it falls below 1.0 — below one expected sample per
                    # batch the path stops being visited at all and no
                    # gradient can restore it (APO, arXiv:2602.05717). Log-y
                    # so the decay reads as a straight line and the approach
                    # to the 1.0 floor is legible: 6ta9hmp6 ran 60.4 (3k) ->
                    # 3.9 (33k), halving every ~8k.
                    "Voluntary-switch rows per batch (absorbing floor = 1.0)",
                    [
                        "player_q_support_vol_switch_rows",
                        "player_q_support_forced_switch_rows",
                    ],
                    log_y=True,
                ),
                lp(
                    # The reference cycle against the switch support it
                    # must hold: voluntary-switch target fraction >= 0.2
                    # is the wire every collapsed lineage tripped.
                    "Reference cycle & switch support",
                    [
                        "player_ref_kl",
                        "player_q_voluntary_switch_target_frac",
                        "player_reg_snapped",
                    ],
                ),
            ],
        ),
        ws.Section(
            # 2026-08-19 reframing: a negative MEAN switch/move gap is the
            # expected sign under correct play (switching spends a turn),
            # so collapse detection lives in the tail — states where the
            # critic actually prefers the switch. Conditioned on the STATE
            # (critic flag), not the taken action, dodging the
            # chosen-switch selection bias of the Aug-15 crossover read.
            name="1.8 · Pivotal-state switch decisions",
            is_open=True,
            panels=[],
        ),
        ws.Section(
            # Does the switch modality's signal actually reach the
            # learner? Both readouts exist because the global staleness
            # instruments are structurally blind to a rare modality:
            # the actor-KL feeding the replay reuse controller is an
            # expectation over the policy, and the capacity probe grades
            # VALUE error, not action-distribution fidelity.
            name="1.9 · Modality-resolved staleness & attenuation",
            is_open=True,
            panels=[
                lp(
                    # The de-averaged k3 actor KL. The _own variant is
                    # the controller's set-point (target 0.045); if the
                    # switch split runs far above it while the global
                    # mean stays ~0.002, the controller is being held
                    # quiet by dilution, not by health.
                    "Actor KL by taken modality vs controller set-point",
                    [
                        "player_learner_actor_forward_kl_switch",
                        "player_learner_actor_forward_kl_move",
                        "player_learner_actor_forward_kl_own",
                    ],
                    log_y=True,
                ),
                lp(
                    # isr = pi_target/mu_actor, the factor v-trace and
                    # Retrace multiply TD errors by. Explore rows record
                    # the TEMPERED log_prob, so mu carries more switch
                    # mass than pi — switch-taken rows sit below 1 and
                    # get heard more faintly as the collapse deepens.
                    # Correct weighting, but a self-reinforcing loop.
                    "Importance ratio by taken modality",
                    [
                        "player_isr_switch_voluntary",
                        "player_isr_switch_forced",
                        "player_isr_move",
                    ],
                ),
                lp(
                    # Cleaner than the mean — isr is heavy-tailed on the
                    # upside. A widening gap = switch evidence being
                    # progressively down-weighted relative to moves.
                    "Fraction of rows with isr < 1 (attenuated)",
                    [
                        "player_isr_below1_switch_voluntary",
                        "player_isr_below1_move",
                    ],
                    range_y=(0, 1),
                ),
                lp(
                    # Context: the reuse cap the controller is holding,
                    # and the global upside-clip fraction.
                    "Replay reuse cap & rho clip fraction",
                    ["player_replay_max_reuses", "player_rho_clip_frac"],
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
                    "Gradient / param norm",
                    ["player_gradient_norm", "player_param_norm"],
                ),
            ],
        ),
        ws.Section(
            name="3 · League",
            is_open=True,
            panels=[
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
                    chart_strings={"title": "league payoff table (row beats column)"},
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
                        "player_loss_entropy",
                        "player_loss_kl",
                        "player_loss_v_win",
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
                    # The one critic's R2 against the v-trace win targets.
                    # The all/private/public ladder went with the
                    # privileged rung (2026-08-25) — its final readings are
                    # in docs/qva-redesign-step0-reference.md.
                    "Value R2",
                    ["player_value_head_r2"],
                    range_y=(-1, 1),
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
                        "player_policy_head_gradient_norm",
                        "player_v_head_gradient_norm",
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
                    # training_step (player_state.step_count) is the
                    # lineage's own counter; lifetime_step is monotonic
                    # across resumes. Plotted together, a params-mode
                    # reload shows as training_step dropping while the
                    # x-axis keeps climbing.
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
        # training_step = player_state.step_count) — the latter restarts
        # on a params-mode reload, which would draw a sawtooth/overdraw.
        # lifetime_step is carried across resumes by construction
        # (rl/online/training/run_state.py's RunState) and is
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
