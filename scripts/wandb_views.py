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
            # NEED-TO-KNOW ONLY: is this run winning, is it healthy, is it
            # collapsing, is the critic calibrated, is it about to OOM.
            # Everything else is drill-down detail in the sections below.
            # 2026-08-30 redesign collapsed 16 sections -> 10 and pulled the
            # canonical copy of every metric that used to be duplicated
            # across 3+ sections up here (scripts/wandb_views.py history —
            # see the old panel list in git log if you need the pre-redesign
            # layout back).
            name="0 · At a glance",
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
                lp(
                    # player_update_skipped is the non-finite gate — a
                    # poisoned update is permanent and the next periodic
                    # save overwrites the last good checkpoint with it
                    # (CLAUDE.md §8), so this is checkpoint protection, not
                    # just a numerics footnote. Never surfaced before this
                    # redesign.
                    "Loss & non-finite gate",
                    ["player_loss", "player_update_skipped"],
                ),
                lp(
                    # THE collapse watch panel — with the adaptivity
                    # controller removed (2026-08-13) and the entropy-floor
                    # dual controllers removed (2026-08-30), modality
                    # collapse has no automated backstop, only these
                    # eyes-on axes (1330 died at modality entropy 0.08;
                    # 1328 gained strength at 0.18-0.26).
                    "Collapse watch: entropy axes & switch rate",
                    [
                        "switch_ratio",
                        "player_action_normalized_entropy",
                        "player_normalized_modality_entropy",
                    ],
                ),
                lp(
                    # R2 of expectations vs v-trace targets — repeated in
                    # detail in "3 · Critic quality & value" alongside value
                    # loss and calibration.
                    "Value R2 (main head)",
                    [
                        # THE privileged-premise discriminator (2026-09-01):
                        # priv >= deploy from 20k is the gate; priv < deploy
                        # sustained past 30k is the abort (the 2026-08-25
                        # falsification re-run on its own instrument).
                        "player_value_head_r2",
                        "player_priv_value_head_r2",
                    ],
                    range_y=(-1, 1),
                ),
                lp(
                    "Process RSS (MB)",
                    ["diag_rss_mb", "diag_node_rss_mb"],
                    smooth=0,
                ),
            ],
        ),
        ws.Section(
            # THE policy gradient since 2026-08-26: NashPG
            # (arXiv:2510.18183) — a PPO-clipped surrogate on the taken
            # action's pi/mu ratio over the batch-normalised v-trace
            # advantage, plus a differentiated forward KL magnet to the
            # periodically snapped reference and an entropy bonus inside
            # the same bracket. The entropy/switch-rate abort watch lives
            # in "0 · At a glance" now — not repeated here.
            name="1 · Policy loss (NashPG)",
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
                    # Per-axis normalised entropies, OBSERVERS since
                    # 2026-08-30: macro dying while the joint H holds is
                    # the modality-collapse shape the global panel in
                    # "0 · At a glance" cannot see.
                    "Per-axis normalised entropies (observers)",
                    [
                        "player_entropy_macro",
                        "player_entropy_micro_taken",
                    ],
                ),
                lp(
                    # Modality decomposition of the throttle on any
                    # taken-action update: per-cell pi mass, as a
                    # switch/move ratio. A falling ratio is the starvation
                    # signature.
                    "Starvation watch (ratio, switch/move)",
                    ["player_policy_prob_ratio"],
                    log_y=True,
                ),
                lp(
                    "Starvation factors: per-cell pi",
                    [
                        "player_policy_prob_switch",
                        "player_policy_prob_move",
                    ],
                    log_y=True,
                ),
                lp(
                    # The zero-avoiding term on the MODALITY MARGINAL:
                    # forward KL from uniform over live modalities, the one
                    # force in the bracket that is not pi-prefactored and so
                    # the only one still acting on an abandoned modality --
                    # identically silent within a modality since 2026-08-31
                    # (the sp75c row form flattened WHICH-move). Read
                    # against prob_switch -- rising mass with this falling
                    # is the term doing its job and relaxing; pinned with
                    # mass unmoved is paying and buying nothing.
                    "Zero-avoiding KL & switch mass",
                    ["player_loss_modality_kl", "player_policy_prob_switch"],
                ),
                lp(
                    "Loss components",
                    [
                        "player_loss_pg",
                        "player_loss_entropy",
                        "player_loss_modality_kl",
                        "player_loss_kl",
                        "player_loss_v_win",
                    ],
                ),
                lp("NLL sum", ["player_nll_sum"]),
            ],
        ),
        ws.Section(
            # What is left of the critic section after the advantage head
            # retired (2026-08-29) and the one-step-label panels went with
            # the last of the Q machinery (2026-08-30), merged with Step 1
            # of docs/critic-weakness-analysis.md (2026-08-23): the per-row
            # JOINT statistics that judge every later step, from the
            # completed-game outcome carried on every chunk. NaN where a
            # batch has no rows in the slice (wandb skips them).
            name="2 · Switch & critic evidence",
            is_open=True,
            panels=[
                lp(
                    # How much gradient switch cells actually receive. The
                    # policy loss trains only the taken action, so
                    # voluntary frac -> 0 IS the starvation mechanism in the
                    # flesh, whatever the head looks like.
                    "Training coverage by modality",
                    [
                        "player_taken_switch_frac",
                        "player_taken_voluntary_switch_frac",
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
                    # Rows of the trunk's OUTPUT converging to one direction
                    # (Noci et al. 2022 rank collapse): cosine rising toward
                    # 1 / participation falling toward 1 is the alarm
                    # (> 0.9 / < 4 pre-registered); ckpt_00182000 read
                    # 0.173 / 10.9 offline, and the first live points
                    # after that restart must match.
                    "Trunk row homogeneity",
                    ["player_trunk_row_cosine", "player_trunk_row_participation"],
                ),
                lp(
                    # The 2026-09-01 opponent-code leaves against their
                    # known init (all lecun 0.0625). Still there tens of
                    # thousands of steps in = never trained: the code is a
                    # random hash and the belief head is learning it.
                    "Opp code: drift from init",
                    [
                        "player_opp_code_logits_rms",
                        "player_opp_code_embedding_rms",
                        "player_belief_head_out_rms",
                        "player_species_belief_rms",
                    ],
                ),
                lp(
                    # Pre-clip grad norms on the same leaves. The embedding
                    # gets gradient on ONE row per (mon, group) through the
                    # straight-through argmax, so a dead group reads as one
                    # row absorbing it -- read beside code_perplexity_min.
                    "Opp code: gradient norms",
                    [
                        "player_opp_code_logits_grad_norm",
                        "player_opp_code_embedding_grad_norm",
                        "player_belief_head_gradient_norm",
                    ],
                ),
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
                    "V outcome R²: all / phase / after switch vs move",
                    [
                        "player_v_outcome_r2_all",
                        "player_v_outcome_r2_early",
                        "player_v_outcome_r2_mid",
                        "player_v_outcome_r2_late",
                        "player_v_outcome_r2_prev_switch",
                        "player_v_outcome_r2_prev_move",
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
                    # Storage-level support: fraction of stored chunks
                    # holding at least one voluntary switch.
                    "Voluntary-switch chunk fraction",
                    ["player_chunk_vol_switch_frac"],
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
                        "player_vol_switch_rows",
                        "player_forced_switch_rows",
                    ],
                    log_y=True,
                ),
                lp(
                    # The magnet/reference cycle against the switch support
                    # it must hold: KL(pi || pi_reg) sawtooths up against
                    # the FROZEN reference, ~0 at each snap (a level
                    # climbing ACROSS snaps is a policy outrunning the snap
                    # period); voluntary-switch target fraction >= 0.2 is
                    # the wire every collapsed lineage tripped.
                    "Reference cycle & switch support",
                    [
                        "player_ref_kl",
                        "player_taken_voluntary_switch_frac",
                        "player_reg_snapped",
                    ],
                ),
            ],
        ),
        ws.Section(
            # The opponent-belief SSL pair (2026-09-01): the discrete code
            # the secret rows carry (grounded by the privileged value CE)
            # and the belief head predicting it from public rows. Split out
            # of the critic section so the mechanism has its own address.
            name="3 · Opponent belief (SSL) & code",
            is_open=True,
            panels=[
                lp(
                    # CE from public rows to the sg'd code, mean over
                    # groups. Falling = beliefs sharpening.
                    "Belief loss",
                    ["player_loss_belief"],
                ),
                lp(
                    # Per-group argmax accuracy (floor 1/16) and the
                    # fraction of mons with an aligned public row (the
                    # label supply).
                    "Belief accuracy & label supply",
                    [
                        "player_belief_accuracy",
                        "player_belief_matched_frac",
                    ],
                ),
                lp(
                    # The accuracy made honest (2026-09-02): the majority
                    # rate a constant predictor scores on the SAME rows,
                    # and accuracy minus it. Above-marginal ~0 = the head
                    # has learnt the batch marginal and nothing else
                    # (a collapsed group is predicted at 100% for free).
                    "Belief accuracy above marginal",
                    [
                        "player_belief_accuracy_above_marginal",
                        "player_belief_majority_rate",
                    ],
                ),
                lp(
                    # The species-only matched control (2026-09-02): a
                    # table keyed on the public row's species, scored on
                    # the same labels and rows. Gain = belief minus
                    # species accuracy: > 0.05 = the head reads public
                    # evidence beyond species; <= 0 = a species lookup.
                    # Read only once the table's own accuracy plateaus
                    # (~2-5k steps after it lands).
                    "Belief gain over species control",
                    [
                        "player_belief_gain_over_species",
                        "player_species_belief_accuracy",
                        "player_species_belief_accuracy_above_marginal",
                    ],
                ),
                lp(
                    "Species control loss",
                    ["player_loss_species_belief"],
                ),
                lp(
                    # Opponent-code usage perplexity per group: min pinned
                    # at 1 = a dead group = the code is ungrounded there
                    # (the collapse instrument for the Dreamer code).
                    "Opponent code perplexity",
                    [
                        "player_code_perplexity_mean",
                        "player_code_perplexity_min",
                    ],
                ),
                lp(
                    # Fraction of value-masked steps carrying any live
                    # opponent row -- the wire supply for all of the above.
                    "Opponent row supply",
                    ["player_code_row_frac"],
                ),
            ],
        ),
        ws.Section(
            # Observer critic quality. The policy no longer reads a Q stack
            # (retired 2026-08-26/30; its link to return is the v-trace
            # advantage), but an action-flat critic still voids the matched
            # control and the starvation discriminators above.
            name="4 · Critic quality & value",
            is_open=True,
            panels=[
                lp(
                    # Both critics' CE against the SAME v-trace win targets
                    # (deployable = matched control, privileged = the
                    # estimator under player_privileged_targets).
                    "Value loss (deploy vs privileged)",
                    ["player_loss_v_win", "player_loss_v_win_priv"],
                ),
                lp(
                    # Mean |priv - deploy| expectation: the 2026-08-25
                    # "worth 0.005 value units" number, re-measured live.
                    "Privileged value gap",
                    ["player_priv_value_gap"],
                ),
                lp(
                    # Directed-message sanity: fraction of valid history
                    # steps with an identified SOURCE row (expect >> 0.5).
                    "History src fraction",
                    ["player_history_src_frac"],
                ),
                lp(
                    # Fresh-row calibration. Was framed as "Q fresh/replay
                    # vs V fresh" pre-2026-08-30 — the Q side retired with
                    # the Q head; only the V-fresh reading remains.
                    "Value R2 calibration (fresh rows)",
                    ["player_value_r2_fresh"],
                ),
                lp(
                    # R2 of expectations vs v-trace targets. Also shown
                    # summarised in "0 · At a glance"; this is the detail
                    # copy beside the rest of the critic reading.
                    "Value R2 (main head)",
                    ["player_value_head_r2"],
                    range_y=(-1, 1),
                ),
                lp(
                    # Pre-clip grad norm per policy-head subtree, the
                    # policy pathway's own gradient scale (the retired
                    # Q-head pair stayed calm through both dx65cpwp
                    # failures).
                    "Policy head: grad norm by subtree",
                    ["player_policy_head_gradient_norm"],
                ),
                lp(
                    "Value expectation",
                    ["value_expectation_mean", "value_expectation_early_mean"],
                ),
                lp(
                    "Win returns",
                    ["player_win_returns_sum", "player_win_returns_min"],
                ),
            ],
        ),
        ws.Section(
            # Does the switch modality's signal actually reach the
            # learner, and is the trust region behaving. Both readouts
            # exist because the global staleness instruments are
            # structurally blind to a rare modality: the actor-KL feeding
            # the replay reuse controller is an expectation over the
            # policy, and the capacity probe grades VALUE error, not
            # action-distribution fidelity.
            name="5 · Staleness, ISR & trust region",
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
                    "Actor KL (ceiling 0.045)",
                    [
                        "player_learner_actor_backward_kl",
                        "player_learner_actor_forward_kl",
                    ],
                ),
                lp(
                    "Replay reuse (controller & cap)",
                    ["player_replay_realised_ratio", "player_replay_max_reuses"],
                ),
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
            name="6 · League",
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
                    # The heatmap above is a snapshot; this is the trend.
                    # league_main_v_{label}_winrate is dynamically keyed per
                    # opponent (step-numbered snapshots AND br-{step} BR
                    # probes both land here), so a regex panel is the only
                    # way to see it over time — this is also where the
                    # project's only ground-truth exploitability read (the
                    # BR probe curve) becomes visible on the dashboard.
                    "League winrate trend (snapshots & BR probes)",
                    None,
                    regex=r"league_main_v_.*_winrate",
                ),
            ],
        ),
        ws.Section(
            name="7 · Behaviour & environment",
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
            name="8 · Gradient norms by module",
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
                lp(
                    "Gradient / param norm (aggregate)",
                    ["player_gradient_norm", "player_param_norm"],
                ),
            ],
        ),
        ws.Section(
            name="9 · Throughput & compile",
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
                lp(
                    # Which (chunk_rows, history_rows) combo of
                    # player_shape_lattice a batch hit — relevant given the
                    # shape-lattice OOM-guard history (CLAUDE.md §1): a
                    # surprise top-bucket compile is what killed three runs
                    # before the lattice was enumerated up front.
                    "Shape lattice combo (T, H)",
                    ["player_shape_T", "player_shape_H"],
                    smooth=0,
                ),
            ],
        ),
        ws.Section(
            # Fed by learner.py's _log_memory_diagnostics (main-only, every
            # memory_diag_interval steps) plus the service's own 10s
            # process.memoryUsage() write — see index.ts:writeMemoryStats.
            # Process RSS is summarised in "0 · At a glance"; not repeated
            # here.
            name="10 · Memory",
            panels=[
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
                lp("Margin std (train batch)", ["margin_std_mean"]),
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
