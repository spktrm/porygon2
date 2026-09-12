"""Build the saved wandb workspace views for this project's dashboards.

Creates/refreshes two views:
  - pokemon-rl         -> "Signal health"  (training-run diagnostics)
  - pokemon-rl-offline -> "Critic health"  (offline critic / Phi ensemble)

Panel keys mirror what rl/online/main.py and rl/offline/train.py log; when
metrics are added or renamed, update the sections here and re-run. Each run without
an --update-url SAVES A NEW VIEW (the API matches by internal id, not
display name); superseded copies are then PRUNED automatically — after each
save, every other view in the project with the SAME display name is
deleted. Personal workspaces ("<user>'s workspace") and any differently
named views are never touched. Pass --keep-old-views to skip pruning.

Usage:
    python scripts/wandb_views.py [--entity ENTITY] [--project rl|offline|both]
        [--update-rl-url URL] [--update-offline-url URL]
        [--keep-old-views]

Requires `pip install wandb-workspaces` and a logged-in wandb credential.
"""

import argparse
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

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


# The eval slate (rl/online/main.py): two slots against the simple
# heuristic, both the EMA params at T=1. `plain-t1` samples the policy
# exactly as the training actors do; `thresholded` samples it with every
# legal cell below player_prune_threshold removed and the rest renormalised
# (HeadParams.prune_threshold) -- the distribution the learner's v-trace
# ratios are built on. Their gap prices the threshold in play. Series are
# keyed by the eval thread's name.
SH = "EvalActor-simpleheuristic"
PLAIN_T1_ACTOR = f"{SH}-plain-t1-0"
THRESHOLDED_ACTOR = f"{SH}-thresholded-1"
EVAL_ACTORS = (PLAIN_T1_ACTOR, THRESHOLDED_ACTOR)


def threshold_winrate_panel():
    """One canonical comparison in the overview and eval-slate sections:
    the policy as sampled by the training actors against the same policy
    thresholded at sampling. Actor-side averages (200-game half-life,
    reset on restart), no additional UI smoothing."""
    return lp(
        "EMA win rate · T=1 plain vs thresholded (200-game half-life)",
        [f"smoothed-wr-{actor}" for actor in EVAL_ACTORS],
        x="lifetime_step",
        smooth=0,
        range_y=(0, 1),
    )


def rl_sections():
    return [
        ws.Section(
            # NEED-TO-KNOW ONLY: is this run winning, is it healthy, is it
            # collapsing, is the critic calibrated, is it about to OOM.
            # Everything else is drill-down detail in the sections below.
            # This section holds the canonical copy of every metric that
            # would otherwise be duplicated across several sections.
            name="0 · At a glance",
            is_open=True,
            panels=[
                threshold_winrate_panel(),
                lp(
                    "EMA alive-mon margin · T=1 plain vs thresholded (200-game half-life)",
                    [f"smoothed-margin-{actor}" for actor in EVAL_ACTORS],
                    x="lifetime_step",
                    smooth=0,
                ),
                lp(
                    "Main-parameter payoff · T=1 (sparse games, UI-smoothed)",
                    [f"main-payoff-{actor}" for actor in EVAL_ACTORS],
                    x="lifetime_step",
                    smooth=0.9,
                    range_y=(-1, 1),
                ),
                lp(
                    "Eval games since restart · T=1 plain and thresholded",
                    [f"games-{actor}" for actor in EVAL_ACTORS],
                    x="lifetime_step",
                    smooth=0,
                ),
                lp(
                    # player_update_skipped is the non-finite gate — a
                    # poisoned update is permanent and the next periodic
                    # save overwrites the last good checkpoint with it, so
                    # this is checkpoint protection, not just a numerics
                    # footnote.
                    "Loss & non-finite gate",
                    ["player_loss", "player_update_skipped"],
                ),
                lp(
                    # THE collapse watch panel — with the adaptivity
                    # controller and the entropy-floor dual controllers both
                    # removed, modality collapse has no automated backstop,
                    # only these eyes-on axes.
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
                        # THE privileged-premise discriminator: priv >=
                        # deploy from 20k is the gate; priv < deploy
                        # sustained past 30k is the abort.
                        "player_value_head_r2",
                        "player_priv_value_head_r2",
                        # The pairwise critics (2026-09-12), same target and
                        # mask: the decomposed instrument for the same premise.
                        "player_pair_value_public_r2",
                        "player_pair_value_private_r2",
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
                    # The ABORT instrument for the flat support hinge, on
                    # its own panel: the hinge is silent above tau, so this
                    # should hold its ~.49; rising toward .93 with
                    # ineffective_confident_mass unmoved is the whole-set
                    # revert.
                    "Within-taken-modality normalised entropy (abort instrument)",
                    ["player_entropy_micro_taken"],
                    range_y=(0, 1),
                ),
                lp(
                    # Modality-marginal normalised entropy, OBSERVER since
                    # 2026-08-30: macro dying while the joint H holds is
                    # the modality-collapse shape the global panel in
                    # "0 · At a glance" cannot see.
                    "Modality-marginal normalised entropy (observer)",
                    ["player_entropy_macro"],
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
                    # The flat support hinge (2026-09-09, replacing the
                    # modality-marginal KL): the one restoring force, silent
                    # above tau. Read against switch mass -- falling as mass
                    # returns is the term relaxing; pinned with mass unmoved
                    # is paying and buying nothing. Detail in section 11.
                    "Support hinge & switch mass",
                    ["player_loss_support", "player_switch_mass_choice"],
                ),
                lp(
                    "Loss components",
                    [
                        "player_loss_pg",
                        "player_loss_entropy",
                        "player_loss_support",
                        "player_loss_kl",
                        "player_loss_v_win",
                    ],
                ),
                lp("NLL sum", ["player_nll_sum"]),
            ],
        ),
        ws.Section(
            # The per-row JOINT statistics that judge every later step,
            # from the completed-game outcome carried on every chunk. NaN
            # where a batch has no rows in the slice (wandb skips them).
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
                    # The flat readout's own way to fail: the bilinear is
                    # a two-factor product with ONE zero-init factor. query
                    # must leave 0 within ~200 steps (its gradient is a
                    # rank-1 outer product of live rows); key must leave
                    # lecun 0.0625 shortly after (its gradient is
                    # proportional to query, so it is frozen for exactly
                    # one step). Either still flat at 2k IS the stall.
                    "Action readout: drift from init",
                    [
                        "player_pointer_query_rms",
                        "player_pointer_key_rms",
                        "player_pointer_local_src_rms",
                        "player_pointer_local_tgt_rms",
                        "player_switch_head_rms",
                        "player_switch_query_rms",
                        "player_switch_key_rms",
                        "player_switch_local_tgt_rms",
                        "player_other_head_rms",
                    ],
                ),
                lp(
                    "Trunk projection parameter RMS",
                    [
                        "player_trunk_attn_out_rms",
                        "player_trunk_mlp_out_rms",
                        "player_trunk_register_rms",
                        "player_trunk_register_norm_scale_rms",
                    ],
                ),
                lp(
                    "Trunk and action-head gradient norms",
                    ["player_action_head_grad_norm", "player_trunk_grad_norm"],
                ),
                lp(
                    # Rows of the trunk's OUTPUT converging to one direction
                    # (Noci et al. 2022 rank collapse): cosine rising toward
                    # 1 / participation falling toward 1 is the alarm
                    # (> 0.9 / < 4 pre-registered).
                    "Trunk row cosine similarity",
                    ["player_trunk_row_cosine"],
                    range_y=(-1, 1),
                ),
                lp(
                    "Trunk centred participation ratio",
                    ["player_trunk_row_participation"],
                ),
                lp(
                    # Every row enters at RMS 1 (L2 16 at 256), so a
                    # group's OUTPUT L2 is what the six blocks wrote on it;
                    # a group pinned near 16 is one the trunk does not
                    # revise.
                    "Trunk output row L2 per group",
                    [],
                    regex="^player_trunk_out_row_l2_",
                    log_y=True,
                ),
                lp(
                    # The input norm's per-group channel scale, zero at init
                    # (effective 1 + it): the only route by which the input
                    # scale disparity the norm removed can come back.
                    "Input norm group scale: drift from zero",
                    [],
                    regex="^player_input_norm_scale_rms_",
                ),
                lp(
                    # The output norm's twin (2026-09-11): how each group's
                    # rows are re-sized on the way OUT to the heads.
                    "Output norm group scale: drift from zero",
                    [],
                    regex="^player_output_norm_scale_rms_",
                ),
                lp(
                    # The 2026-09-02 history-encoder leaves. attn_out is
                    # ZERO-init and must leave 0 within ~200 steps (still
                    # flat at 2k = the two-factor stall; fallback is a
                    # lecun attn_out behind a zero-init scalar gate);
                    # query/key start at lecun ~0.044, the slot write gate
                    # at ~0.028.
                    "History encoder: drift from init",
                    [
                        "player_history_step_attn_out_rms",
                        "player_history_step_attn_qk_rms",
                        "player_history_slot_gate_rms",
                        "player_history_step_attn_grad_norm",
                    ],
                ),
                lp(
                    # Realised outcome of voluntary switches minus moves at
                    # matched V(s). Per-batch n is tiny; read smoothed and
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
                    # are taken vs where moves are.
                    "V(s) at voluntary switches vs moves",
                    ["player_mv_v_at_vol_switch", "player_mv_v_at_move"],
                    smooth=0.99,
                ),
                lp(
                    # Outcome calibration of the V head. prev_switch vs
                    # prev_move is the post-switch pessimism read.
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
                    # to the 1.0 floor is legible.
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
            # The eval slate: the policy as the training actors sample it
            # against the same policy thresholded at sampling, both EMA at
            # T=1.
            name="3c · Eval slate · T=1 plain vs thresholded",
            is_open=True,
            panels=[
                threshold_winrate_panel(),
                lp(
                    "Eval games since restart · T=1 plain vs thresholded",
                    [f"games-{actor}" for actor in EVAL_ACTORS],
                    x="lifetime_step",
                    smooth=0,
                ),
                lp(
                    "Voluntary switches per offered decision · T=1 plain vs thresholded",
                    [f"switch-frac-{actor}" for actor in EVAL_ACTORS],
                    x="lifetime_step",
                    smooth=0.95,
                    range_y=(0, 1),
                ),
                lp(
                    "Eval ms per decision · T=1 plain vs thresholded (CPU)",
                    [f"ms-per-step-{actor}" for actor in EVAL_ACTORS],
                    x="lifetime_step",
                    smooth=0.9,
                ),
            ],
        ),
        ws.Section(
            # Observer critic quality. The policy no longer reads a Q stack
            # (its link to return is the v-trace advantage), but an
            # action-flat critic still voids the matched control and the
            # starvation discriminators above.
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
                    # Mean |priv - deploy| expectation, re-measured live.
                    "Privileged value gap",
                    ["player_priv_value_gap"],
                ),
                lp(
                    # The PBRS potential channel (2026-09-11): its std share
                    # of the actor advantage should FALL as the potential
                    # head fits (the head's lag IS the shaping; a floor is
                    # the unfitted part persisting). fit_r2 is the head
                    # against its EXACT target -Phi -- the distance from
                    # inert; the pre-registered read is >= .9 by 20k.
                    "PBRS potential channel",
                    [
                        "player_potential_adv_share",
                        "player_potential_head_fit_r2",
                        "player_potential_head_r2",
                        "player_potential_win_adv_corr",
                    ],
                ),
                lp(
                    # The unit position potential (eta-free, on the wire at
                    # any strength). switch_delta is DESCRIPTIVE;
                    # adv_switch/move split the channel's advantage by the
                    # taken modality; grad_share is the head's part of the
                    # global clip norm.
                    "Position potential",
                    [
                        "player_potential_mean",
                        "player_potential_std",
                        "player_potential_switch_delta_mean",
                        "player_potential_adv_switch",
                        "player_potential_adv_move",
                        "player_potential_head_grad_share",
                    ],
                ),
                lp(
                    # Directed-message sanity: fraction of valid history
                    # steps with an identified SOURCE row (expect >> 0.5).
                    "History src fraction",
                    ["player_history_src_frac"],
                ),
                lp(
                    # The step GAT reading "who did this to me": the mass a
                    # NON-source row places on the step's source rows,
                    # beside what uniform attention would place (the source
                    # rows' share of live rows). Above uniform = the
                    # relation is being read; at uniform with the
                    # normalised entropy pinned at 1.0 = the GAT never
                    # learned to select (read with attn_out rms).
                    "History step attention",
                    [
                        "player_history_step_attn_to_src",
                        "player_history_step_attn_to_src_uniform",
                        "player_history_step_attn_entropy",
                    ],
                ),
                lp(
                    # The backbone's mean slot write gate over touched
                    # steps. Pinned at 0 (nothing written) or 1 (memory
                    # overwritten every step) is the collapse shape;
                    # pre-registered band (0.1, 0.9).
                    "History write gate",
                    ["player_history_gate_mean"],
                ),
                lp(
                    # Fresh-row calibration, the V-fresh reading.
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
                    # policy pathway's own gradient scale.
                    "Action-head gradient norm",
                    ["player_action_head_gradient_norm"],
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
            name="4b · Pair value heads",
            is_open=True,
            panels=[
                lp(
                    "Pair value R2 vs CLS critics",
                    [
                        "player_pair_value_public_r2",
                        "player_pair_value_private_r2",
                        "player_value_head_r2",
                        "player_priv_value_head_r2",
                    ],
                    range_y=(-1, 1),
                ),
                lp(
                    "Pair value: gradient norms",
                    [
                        "player_pair_value_public_gradient_norm",
                        "player_pair_value_private_gradient_norm",
                    ],
                ),
                # The pairwise entity critics (2026-09-12): two generalised
                # additive models over 12 entity rows (unary + antisymmetric
                # cross pair + symmetric same-side pair, softmax weights),
                # both over POST-trunk rows (public rows; my sheet rows and
                # the opponent-truth rows). Their R2 also sits on "Value R2
                # (main head)" beside the CLS critics.
                lp(
                    # MSE against the same scalar v-trace return the CLS
                    # critics' two-hot is built from.
                    "Pair value loss",
                    ["player_loss_pair_value_public", "player_loss_pair_value_private"],
                ),
                lp(
                    # Variance of each part over the variance of V (need not
                    # sum to 1). Pre-registered: the cross share leaves the
                    # floor while R2 rises, else the value lives in the
                    # unary terms and the pre-trunk pair term is the fallback.
                    "Pair value: part shares",
                    None,
                    regex="^player_pair_value_(public|private)_share_",
                ),
                lp(
                    # Signed means of the five parts (unary mine/theirs,
                    # cross, synergy mine/theirs): where the value sits.
                    "Pair value: partials",
                    None,
                    regex="^player_pair_value_(public|private)_partial_",
                ),
                lp(
                    # Normalised entropy of the softmax pair weights (1 =
                    # uniform over alive pairs, 0 = one pair) -- concentrating
                    # on the decisive matchup is the intended reading, so
                    # this is descriptive.
                    "Pair value: weight entropy",
                    None,
                    regex="^player_pair_value_(public|private)_("
                    "cross_weight_entropy_norm|synergy_weight_entropy_norm)$",
                    range_y=(0, 1),
                ),
                lp(
                    "Pair value: unary cancellation",
                    None,
                    regex="^player_pair_value_(public|private)_unary_cancellation$",
                ),
                lp(
                    # Mean |m| / |s| over alive pairs: the pair terms'
                    # magnitude in [0, 1].
                    "Pair value: pair magnitudes",
                    None,
                    regex="^player_pair_value_(public|private)_(cross|synergy)_abs_mean$",
                ),
                lp(
                    # The four-questions drift panels: each pair function's
                    # query (0 at init) and key (0.0625 at init) rms.
                    "Pair value: kernel rms",
                    None,
                    regex="^player_pair_value_(public|private)_.*_(query|key)_rms$",
                ),
                lp(
                    "Pair value: applied delta rms",
                    None,
                    regex="^player_applied_delta_rms_pair_value_",
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
                    # Compare the taken-modality splits with the live global
                    # forward KL. The retired _own key is not a set-point.
                    "Actor forward KL · switches, moves and global",
                    [
                        "player_learner_actor_forward_kl_switch",
                        "player_learner_actor_forward_kl_move",
                        "player_learner_actor_forward_kl",
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
                    # rho_clip_frac reads the THRESHOLDED target/behaviour
                    # ratio since 2026-09-09; its _raw twin is the series
                    # comparable to before the restart.
                    "Clip fractions",
                    [
                        "player_impact_clip_frac",
                        "player_rho_clip_frac",
                        "player_rho_clip_frac_raw",
                    ],
                ),
                lp(
                    # Target/behaviour ratio ESS, thresholded vs raw twin.
                    "ISR ESS · v-trace ratio (thresholded vs raw)",
                    ["player_isr_ess", "player_isr_ess_raw"],
                    range_y=(0, 1),
                ),
                lp(
                    # The LEARNER/behaviour ratio -- a different population
                    # from the v-trace ratio above; never one axis.
                    "Learner/behaviour ratio ESS",
                    ["player_learner_actor_ess"],
                    range_y=(0, 1),
                ),
                lp(
                    "Learner/behaviour ratio tail mass above 2",
                    ["player_learner_actor_ratio_tail_gt2"],
                    range_y=(0, 1),
                ),
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
                # titles and a diverging win-rate colour scale.
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
                    # Whole-game length off terminal chunks' done rows;
                    # there is no request cap, so this is the distribution
                    # to watch.
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
                        "player_action_head_gradient_norm",
                        "player_v_head_gradient_norm",
                        "player_pair_value_public_gradient_norm",
                        "player_pair_value_private_gradient_norm",
                    ],
                    log_y=True,
                ),
                lp(
                    # Pre-clip grad norms of the deployable heads and the
                    # trunk (telemetry._GRAD_SUBTREES): the value head's
                    # beside the readout's and the trunk's.
                    "Pre-clip grad norms: readout, value head, trunk",
                    [
                        "player_action_head_grad_norm",
                        "player_value_head_grad_norm",
                        "player_trunk_grad_norm",
                        "player_history_step_attn_grad_norm",
                    ],
                    log_y=True,
                ),
                lp(
                    "Gradient / param norm (aggregate)",
                    ["player_gradient_norm", "player_param_norm"],
                ),
                lp(
                    # Column-block rms of the three state kernels (public
                    # persistent / transient, private) by feature group,
                    # against "other" (level, gender, item effect, pp, tera,
                    # volatiles, stats).
                    "State kernel rms by feature group",
                    [
                        "player_state_kernel_rms_hp",
                        "player_state_kernel_rms_status",
                        "player_state_kernel_rms_boosts",
                        "player_state_kernel_rms_other",
                    ],
                ),
                lp(
                    # What Adam APPLIED to the readout leaves a support
                    # force acts on (post-clip, post-revert rms): the switch
                    # pair's query and ally-side scalar beside the move
                    # pair's.
                    "Applied update rms · action readout leaves",
                    [
                        "player_applied_delta_rms_switch_query",
                        "player_applied_delta_rms_switch_local_tgt",
                        "player_applied_delta_rms_pointer_query",
                        "player_applied_delta_rms_pointer_key",
                        "player_applied_delta_rms_pointer_local_tgt",
                    ],
                    log_y=True,
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
                    # player_shape_lattice a batch hit — a surprise
                    # top-bucket compile is the OOM-guard failure the
                    # enumerated lattice exists to prevent.
                    "Shape lattice combo (T, H)",
                    ["player_shape_T", "player_shape_H"],
                    smooth=0,
                ),
            ],
        ),
        ws.Section(
            # Fed by the ActorStats sink (rl/environment/actor_stats.py):
            # every training actor, its env and the InferenceServer record
            # wall-time per phase, drained every actor_stats_log_steps as
            # means over the pool. Where an actor's step goes — the
            # system rate is actor-bound.
            name="9b · Actor step timing",
            panels=[
                lp(
                    # service_wait includes the OPPONENT actor's whole
                    # step in self-play (both sides choose before the sim
                    # advances) — "waiting on the game server", not
                    # service CPU. other = step_total minus the four.
                    "Actor step decomposition (ms)",
                    [
                        "actor_time_step_total",
                        "actor_time_service_wait",
                        "actor_time_process_state",
                        "actor_time_history_clip",
                        "actor_time_inference",
                        "actor_time_other",
                    ],
                ),
                lp(
                    "CPU actor forward time (ms)",
                    ["actor_infer_forward"],
                ),
                lp(
                    "CPU actor history bucket level",
                    ["actor_infer_history_level"],
                ),
                lp(
                    # actor = pool aggregate of env steps; learner = the
                    # SYSTEM rate (learner steps per wall second over the
                    # drain interval). The speed comparison reads both
                    # against the carry-OFF window on the same code.
                    "Steps/sec: actors (pool) & learner (system rate)",
                    ["actor_steps_per_sec", "learner_steps_per_sec"],
                ),
                lp(
                    # The carry path's own read: steps / packed rows of
                    # the suffix a request actually sends, against the
                    # 64/128+ a full window pads to.
                    "History carry: suffix size per request",
                    ["actor_history_suffix_steps", "actor_history_suffix_rows"],
                ),
                lp(
                    # Fraction of requests recomputed from h0, by reason.
                    # game_start ~ 1/60; rewrite = an Illusion |replace|
                    # rewrote past rows; gap = the window no longer holds
                    # the carried step. Total > 0.1 = a continuity bug,
                    # not a tuning question.
                    "History carry: recompute fraction",
                    [
                        "actor_history_recompute_frac",
                        "actor_history_recompute_game_start",
                        "actor_history_recompute_rewrite",
                        "actor_history_recompute_gap",
                    ],
                ),
            ],
        ),
        ws.Section(
            # Fed by log_memory_diagnostics in
            # rl/online/training/diagnostics.py (main-only, every
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
        ws.Section(
            # The 2026-09-09 support set read top to bottom: what the policy
            # exposes (flat legal-cell support), what the hinge asks and
            # pays, what the v-trace threshold discards and the trace it
            # cuts. Every trajectory keyed to lifetime_step.
            name="11 · Action support & exposure",
            is_open=True,
            panels=[
                lp(
                    # The floor of the policy's legal support, as flat
                    # complete actions: the hinge lifts min toward tau.
                    "Legal-cell probability floor (min / median, log)",
                    ["player_support_min_prob", "player_support_median_prob"],
                    log_y=True,
                ),
                lp(
                    "Legal cells below the lines (fraction)",
                    [
                        "player_support_frac_below_p01",
                        "player_support_frac_below_p005",
                        "player_support_frac_below_p001",
                    ],
                    range_y=(0, 1),
                ),
                lp(
                    "Switch-cell floor vs move-cell floor (log)",
                    [
                        "player_support_switch_min_prob",
                        "player_support_move_min_prob",
                    ],
                    log_y=True,
                ),
                lp(
                    "Switch vs move cells below .005 (fraction)",
                    [
                        "player_support_switch_frac_below_p005",
                        "player_support_move_frac_below_p005",
                    ],
                    range_y=(0, 1),
                ),
                lp(
                    "Legal cells per decision",
                    ["player_support_legal_count"],
                ),
                lp(
                    # The hinge: loss, the fraction of legal cells it is
                    # pushing on, the per-row mass it asks (N * tau_row) and
                    # how often the tau_max_mass clamp binds.
                    "Support hinge loss & active fraction",
                    ["player_loss_support", "player_support_active_fraction"],
                ),
                lp(
                    "Support ask N * tau_row & clamp saturation",
                    ["player_support_n_tau_row", "player_support_saturated_frac"],
                    range_y=(0, 1),
                ),
                lp(
                    # The discard rate: the revert trigger is > 1% of taken
                    # rows sustained over a 250k-fresh-decision window.
                    "v-trace discard rate · taken rows, by taken modality",
                    [
                        "player_discard_taken_frac",
                        "player_discard_taken_frac_switch",
                        "player_discard_taken_frac_move",
                    ],
                    range_y=(0, 0.05),
                ),
                lp(
                    "v-trace discard · legal cells below the line, position of discards",
                    ["player_discard_legal_frac", "player_discard_position_mean"],
                    range_y=(0, 1),
                ),
                lp(
                    # c is RAW (rho only carries the threshold, the
                    # restriction the 2026-09-09 cut audit fired), so _raw is
                    # the live trace and the thresholded series the cut a
                    # thresholded c would have made.
                    "Trace length · live (raw c) vs had c been thresholded",
                    ["player_trace_len_mean", "player_trace_len_mean_raw"],
                ),
                lp(
                    "Directional switch-logit gradient by term",
                    [
                        "player_switch_logit_grad_pg",
                        "player_switch_logit_grad_entropy",
                        "player_switch_logit_grad_magnet",
                        "player_switch_logit_grad_support",
                        "player_switch_logit_grad_actor_total",
                    ],
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
        # The SDK wraps the URL token in nw-...-v when loading. Accept
        # internal names too, without saving a doubly wrapped view name.
        parsed_url = urlparse(update_url)
        query = parse_qs(parsed_url.query)
        view_name = query.get("nw", [""])[0]
        if view_name.startswith("nw-") and view_name.endswith("-v"):
            query["nw"] = [view_name[3:-2]]
            update_url = urlunparse(
                parsed_url._replace(query=urlencode(query, doseq=True))
            )
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
    # Confirm the saved URL resolves before allowing stale-view pruning.
    saved_workspace = ws.Workspace.from_url(workspace.url)
    if saved_workspace._internal_id != workspace._internal_id:
        raise RuntimeError("Saved workspace URL resolves to a different view")
    print(f"{project} view: {workspace.url}")
    return workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="jtwin")
    parser.add_argument(
        "--project",
        choices=("rl", "offline", "both"),
        default="both",
        help="Select saved views to refresh; defaults to both projects.",
    )
    parser.add_argument("--update-rl-url", default=None)
    parser.add_argument("--update-offline-url", default=None)
    parser.add_argument(
        "--keep-old-views",
        action="store_true",
        help="Skip deleting superseded same-name views after saving.",
    )
    args = parser.parse_args()

    requests = []
    if args.project in ("rl", "both"):
        requests.append(
            (
                "pokemon-rl",
                "Signal health",
                rl_sections(),
                args.update_rl_url,
                ws.WorkspaceSettings(x_axis="lifetime_step"),
                "lifetime_step",
            )
        )
    if args.project in ("offline", "both"):
        requests.append(
            (
                "pokemon-rl-offline",
                "Critic health",
                offline_sections(),
                args.update_offline_url,
                None,
                None,
            )
        )
    for project, name, sections, update_url, settings, force_x in requests:
        workspace = save_view(
            args.entity,
            project,
            name,
            sections,
            update_url,
            settings=settings,
            force_x=force_x,
        )
        if not args.keep_old_views:
            prune_stale_views(args.entity, project, name, workspace._internal_id)


if __name__ == "__main__":
    main()
