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


# The eval slate since 2026-09-09 (rl/online/main.py): two slots against
# the simple heuristic, both the EMA params at T=1. `plain-t1` samples the
# policy exactly as the training actors do; `thresholded` samples it with
# every legal cell below player_prune_threshold removed and the rest
# renormalised (HeadParams.prune_threshold) -- the distribution the
# learner's v-trace ratios are built on. Their gap prices the threshold
# in play. Series are keyed by the eval thread's name, so the earlier
# `-0`/`-1` (T=0.5), `-t1-2` and `-search-3` series end at that restart.
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
            # 2026-08-30 redesign collapsed 16 sections -> 10 and pulled the
            # canonical copy of every metric that used to be duplicated
            # across 3+ sections up here (scripts/wandb_views.py history —
            # see the old panel list in git log if you need the pre-redesign
            # layout back).
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
                    # The ABORT instrument for the flat support hinge
                    # (2026-09-09), on its own panel: the sp75c row-form
                    # uniform KL pinned this at .93 while the control fell
                    # to .84 and halved the exploit. The hinge is silent
                    # above tau, so this should hold its ~.49; rising
                    # toward .93 with ineffective_confident_mass unmoved is
                    # the whole-set revert.
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
                    # (> 0.9 / < 4 pre-registered); ckpt_00182000 read
                    # 0.173 / 10.9 offline, and the first live points
                    # after that restart must match.
                    "Trunk row cosine similarity",
                    ["player_trunk_row_cosine"],
                    range_y=(-1, 1),
                ),
                lp(
                    "Trunk centred participation ratio",
                    ["player_trunk_row_participation"],
                ),
                lp(
                    # 2026-09-10: every row enters at RMS 1 (L2 16 at 256),
                    # so a group's OUTPUT L2 is what the six blocks wrote on
                    # it. Unnormalised, the history rows sat at ~1040 in and
                    # out (moved 2%) while CLS went 2.85 -> 1012; a group
                    # pinned near 16 is one the trunk does not revise.
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
                        "player_revealed_belief_rms",
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
                    # groups. Falling = beliefs sharpening. Since
                    # 2026-09-05 (irqeetfg ~1.15M) the label is the code
                    # of each mon's HIDDEN tokens only, so every belief
                    # panel breaks there: before, the full-sheet code.
                    "Belief loss",
                    ["player_loss_belief"],
                ),
                lp(
                    # Per-group argmax accuracy (floor 1/16), the fraction
                    # of mons with an aligned public row, and of those the
                    # share still carrying a hidden token (the label
                    # supply is their product).
                    "Belief accuracy & label supply",
                    [
                        "player_belief_accuracy",
                        "player_belief_matched_frac",
                        "player_belief_hidden_frac",
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
                    # The revealed-row control: an MLP over the matched
                    # mon's own PRE-trunk public row alone (species, revealed
                    # moves, item, ability, state), same labels and rows.
                    # Margin = belief minus its accuracy = inference from
                    # CONTEXT (history, the other rows); up is good. B3
                    # rule (2026-09-04) FIRED: margin 0.20 -> 0.017 by
                    # 1.15M, the control read the full-sheet label off the
                    # row's own tokens. Under the hidden-token label
                    # (2026-09-05) this control is the positive control:
                    # its above-marginal accuracy must FALL to the
                    # species control's (the row shows nothing the label
                    # encodes), and the head's must stay above both.
                    "Belief context margin over revealed-row control",
                    [
                        "player_belief_context_margin",
                        "player_revealed_belief_accuracy",
                        "player_revealed_belief_accuracy_above_marginal",
                    ],
                ),
                lp(
                    "Control losses",
                    ["player_loss_species_belief", "player_loss_revealed_belief"],
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
                    # The belief LABEL's usage over the rows the loss
                    # scores: the same code net over the hidden tokens
                    # alone. Min pinned at 1 = a dead label; read beside
                    # above-marginal, which is floored by 1/perplexity.
                    "Hidden-token label perplexity",
                    [
                        "player_hidden_code_perplexity_mean",
                        "player_hidden_code_perplexity_min",
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
            # The latent transition model (2026-09-05): g = 2 trunk blocks
            # over the 73 policy-readable post-trunk rows conditioned on
            # the taken cell's readout rows and a chance code z (2 groups
            # x 16 classes; prior from h_t and the cell, posterior also
            # from sg(h_{t+1}), straight-through). Heads on the imagined
            # rows: consistency to the real next rows, grounding to the 21
            # pre-trunk target rows (the old dynamics head's label, in the
            # NEXT step's layout), the SHARED value head and readout, the
            # next legal set + request kind, done. KL balanced 0.5 / 0.1
            # with 1 free nat. Every panel says which decode it reads:
            # posterior sample (what the model can fit given the outcome)
            # or prior mode (what a rollout could imagine).
            name="3b · Transition model",
            is_open=True,
            panels=[
                lp(
                    # The bracket and its terms; every one is 0-best. The
                    # grounding term's copy predictor scores exactly 1.
                    "Transition loss terms",
                    [
                        "player_loss_transition",
                        "player_loss_transition_ground",
                        "player_loss_transition_cons",
                        "player_loss_transition_kl",
                        "player_loss_transition_value",
                        "player_loss_transition_value_k1",
                        "player_loss_transition_kind",
                        "player_loss_transition_termination",
                        "player_loss_transition_decode",
                        "player_loss_transition_generator",
                        "player_loss_transition_align",
                    ],
                ),
                lp(
                    # Nats per transition between posterior and prior --
                    # THE chance-node number: how much of the transition
                    # the code carries that h_t and my action do not.
                    # Gate 0.5-3.0: -> 0 is posterior collapse (a mean
                    # model with a dead code), at the 5.5-nat capacity is
                    # the posterior copying t+1. free_frac = share of
                    # transitions under the 1-nat floor (no gradient).
                    "Transition KL (nats)",
                    [
                        "player_transition_kl",
                        "player_transition_kl_k1",
                        "player_transition_kl_free_frac",
                    ],
                ),
                lp(
                    # KL split by the transition's shape: spanned edges
                    # (short <= 2 vs long >= 4) and whether a matched
                    # opponent row changed an id token. The plan's
                    # prediction is long > short and reveal > no_reveal by
                    # >= 0.2 nats -- the code should carry MORE where more
                    # unobserved things happened.
                    "Transition KL by span / reveal",
                    [
                        "player_transition_kl_short",
                        "player_transition_kl_long",
                        "player_transition_kl_reveal",
                        "player_transition_kl_no_reveal",
                    ],
                ),
                lp(
                    # Code usage from the batch marginals, per group (mean
                    # and the worst group). 1 = a dead group; 16 = uniform.
                    # Gate: post min >= 1.5, prior min >= 1.3. agree = the
                    # prior's argmax matches the posterior's. sample_is_mode
                    # = share of transitions where the DRAWN posterior code
                    # is its argmax in every group: exactly 1.0 means the
                    # learner ran without its sampling rng (argmax decode).
                    "Transition code perplexity",
                    [
                        "player_transition_post_perplexity_mean",
                        "player_transition_post_perplexity_min",
                        "player_transition_prior_perplexity_mean",
                        "player_transition_prior_perplexity_min",
                        "player_transition_prior_post_agree",
                        "player_transition_post_sample_is_mode",
                    ],
                ),
                lp(
                    # Grounding R^2 of the aligned change per row group,
                    # posterior-sample decode: 1 = perfect, 0 = the copy
                    # predictor, negative = worse than copying. The mean
                    # head banked 0.528 / 0.487 / 0.626; gate public >= 0.68.
                    "Transition grounding gain (posterior)",
                    [
                        "player_transition_gain_public",
                        "player_transition_gain_private",
                        "player_transition_gain_field",
                    ],
                ),
                lp(
                    # The same public read from the PRIOR-MODE decode, and
                    # the hp-moved subset both ways. The prior read is
                    # EXPECTED BELOW the mean head's 0.528: one branch of a
                    # bimodal law is further from the truth in MSE than the
                    # mean. Not a regression; the KL panel is where the
                    # branching went.
                    "Transition grounding gain: prior vs posterior",
                    [
                        "player_transition_gain_public",
                        "player_transition_gain_public_prior",
                        "player_transition_gain_hp_moved",
                        "player_transition_gain_hp_moved_prior",
                        "player_transition_hp_moved_frac",
                    ],
                ),
                lp(
                    # Grounding gain on the public rows split by the
                    # transition's shape (posterior decode).
                    "Transition grounding gain by span / reveal",
                    [
                        "player_transition_gain_public_short",
                        "player_transition_gain_public_long",
                        "player_transition_gain_public_reveal",
                        "player_transition_gain_public_no_reveal",
                    ],
                ),
                lp(
                    # Consistency R^2 of the imagined post-trunk rows per
                    # row kind against the real next rows (copy = 0).
                    # Active again at coefficient 1 after the 2026-09-08
                    # validity fix. Both-absent rows are excluded; unscored
                    # groups report 0. Earlier values use a different mask.
                    "Transition consistency gain by row kind",
                    [
                        "player_transition_cons_gain_cls",
                        "player_transition_cons_gain_public_entity",
                        "player_transition_cons_gain_private_entity",
                        "player_transition_cons_gain_move_slot",
                        "player_transition_cons_gain_target_slot",
                        "player_transition_cons_gain_field",
                        "player_transition_cons_gain_history_field",
                        "player_transition_cons_gain_prev_action",
                        "player_transition_cons_gain_info",
                        "player_transition_cons_gain_history_entity",
                    ],
                ),
                lp(
                    # The calibration read search depends on: R^2 of the
                    # SHARED value head on the imagined CLS row against the
                    # t+1 win_returns, beside the same head on real rows.
                    # VACUOUS without the copy baseline (Step 3b correction
                    # 1): value barely moves between requests, so V(h_t)
                    # already scores ~0.9 here. Read the next two panels.
                    # value_gap = |V(imagined) - v_target(t+1)| in support
                    # units, lower is better; _prior is the rollout-side
                    # decode, _switch/_move by the taken modality.
                    "Transition value R2 / gap (raw)",
                    [
                        "player_transition_value_r2",
                        "player_value_head_r2",
                        "player_transition_value_gap",
                        "player_transition_value_gap_prior",
                        "player_transition_value_gap_switch",
                        "player_transition_value_gap_move",
                    ],
                ),
                lp(
                    # Step 3b (2026-09-06): the honest calibration reads,
                    # both scaled so the COPY predictor (imagined = root)
                    # is exactly 0 and the real next state is 1. delta_r2
                    # = uncentred R2 of the imagined CHANGE in V against
                    # the real change; value_gain = (ce_copy - ce_imagined)
                    # / (ce_copy - ce_real). _prior decodes from the
                    # prior's MODE (what search samples from) -- the
                    # number B must move (pre-fix offline: post 0.45 /
                    # prior -0.24 on delta_r2). Gate: value_gain > 0 and
                    # rising; delta_r2 >= gain_hp_moved; _prior >= 0.62 x
                    # posterior. value_gain < 0 sustained 5k = off-manifold
                    # abort (knob value_trains_v_head -> False).
                    "Transition value calibration (copy = 0, real = 1)",
                    [
                        "player_transition_value_delta_r2",
                        "player_transition_value_delta_r2_prior",
                        "player_transition_value_delta_r2_switch",
                        "player_transition_value_delta_r2_move",
                        "player_transition_value_gain",
                        "player_transition_value_gain_prior",
                    ],
                ),
                lp(
                    # The 2026-09-07 splits: delta_r2 on transitions where
                    # a row the trunk zeroed at t EXISTS at t+1 (the rows
                    # the 2026-09-05 imagine could never write; the
                    # pre-registered prediction is that this split closes
                    # toward the other) and the second unroll offset (the
                    # change over TWO steps against copy, through an
                    # imagined intermediate).
                    "Transition value calibration by newly-valid rows / offset",
                    [
                        "player_transition_value_delta_r2_newly_valid",
                        "player_transition_value_delta_r2_no_newly_valid",
                        "player_transition_value_delta_r2_k1",
                        "player_transition_newly_valid_frac",
                        "player_transition_cons_gain_newly_valid",
                    ],
                ),
                lp(
                    # The two sums behind every delta_r2 panel (2026-09-07
                    # audit): a split with few rows per batch (switch)
                    # logs per-batch ratios of magnitude > 100, so the
                    # pooled read over a window is 1 - mean(sse) /
                    # mean(energy) from THESE, never the mean of the
                    # ratio panel.
                    "Transition delta sums (pool: 1 - mean sse / mean energy)",
                    [
                        "player_transition_value_delta_sse",
                        "player_transition_value_delta_energy",
                        "player_transition_value_delta_sse_prior",
                        "player_transition_value_delta_sse_switch",
                        "player_transition_value_delta_energy_switch",
                        "player_transition_value_delta_sse_move",
                        "player_transition_value_delta_energy_move",
                    ],
                ),
                lp(
                    # The three CEs behind value_gain, all against the
                    # t+1 label: the root's V (copy), the imagined V
                    # (loss_transition_value), the prior-mode decode's V,
                    # and the real next state's V (the floor). Lower is
                    # better; imagined between copy and real = a gain.
                    "Transition value CEs",
                    [
                        "player_transition_value_ce_copy",
                        "player_loss_transition_value",
                        "player_transition_value_ce_prior",
                        "player_transition_value_ce_real",
                    ],
                ),
                lp(
                    # The latent action (2026-09-07): the action encoder
                    # under the EXACT decode objective at the root.
                    # decode_acc = expected top-1 decode of the taken cell
                    # among the legal ones (1/num_legal is chance);
                    # action_mi = log(n) - H(A | U) in nats (0 = the
                    # collapsed symmetric encoding, log(n) = every legal
                    # cell its own code); action_entropy = H(q(u | h, a))
                    # per taken cell; perplexity = usage of the DRAWN
                    # codes over the batch (1 = one code for everything,
                    # 64 = uniform); sample_is_mode exactly 1.0 = the
                    # learner ran without its sampling rng; logit_mean
                    # is the shift-invariant direction the objective
                    # leaves free (drifting = the optimiser, not the loss).
                    "Transition latent action (decode)",
                    [
                        "player_transition_decode_acc",
                        "player_transition_action_mi",
                        "player_transition_action_entropy",
                        "player_transition_action_perplexity_mean",
                        "player_transition_action_sample_is_mode",
                        "player_transition_action_num_legal",
                        "player_transition_action_overflow_frac",
                        "player_transition_action_logit_mean",
                        "player_transition_action_logit_std",
                    ],
                ),
                lp(
                    # The candidate generator: KL of its first conditional
                    # to the base policy's latent target (the 64-way
                    # prior; CE - H, so 0 is exact) and of the later
                    # slots to the without-replacement conditionals, at
                    # the real root and at the imagined nodes (_k1, _k2:
                    # the deployed reader on the states search will hand
                    # it); target entropy / perplexity = how many
                    # consequence classes the policy spreads over;
                    # coverage = the target mass the J teacher codes hold
                    # (gate >= 0.9); occupied / support = the teacher's
                    # support size at 0.99 mass.
                    "Transition candidate generator",
                    [
                        "player_transition_generator_kl_first",
                        "player_transition_generator_kl_later",
                        "player_transition_generator_kl_first_k1",
                        "player_transition_generator_kl_first_k2",
                        "player_transition_generator_target_entropy",
                        "player_transition_generator_target_perplexity",
                        "player_transition_generator_coverage",
                        "player_transition_generator_occupied",
                        "player_transition_generator_support",
                    ],
                ),
                lp(
                    # Alignment: the encoder on the IMAGINED state with the
                    # recorded cell against its real-state distribution,
                    # KL per node (0 = the code keeps its meaning after
                    # one / two imagined steps).
                    "Transition action-code alignment on imagined nodes",
                    [
                        "player_transition_align_kl_k1",
                        "player_transition_align_kl_k2",
                        "player_loss_transition_align",
                    ],
                ),
                lp(
                    # The next request kind (gate >= 0.95), done, and the
                    # conditional terminal outcome: terminal_ce / _acc on
                    # the actual terminal successors only (the reader the
                    # fractional-continuation backup needs),
                    # terminal_payoff_err = |(1 - c) T - recorded reward|
                    # over every eligible row (0 on non-terminal rows is
                    # the recorded reward there), terminal_rows the supply.
                    "Transition kind / done / terminal outcome",
                    [
                        "player_transition_kind_acc",
                        "player_transition_kind_acc_k1",
                        "player_transition_done_acc",
                        "player_transition_done_acc_k1",
                        "player_transition_terminal_ce",
                        "player_transition_terminal_acc",
                        "player_transition_terminal_payoff_err",
                        "player_transition_terminal_rows",
                        "player_transition_done_frac",
                    ],
                ),
                lp(
                    # Spanned engine steps per transition (mean / p90) and
                    # the share with an opponent reveal -- the data
                    # geometry the split panels condition on. Also the row
                    # supply: transitions with a valid t+1 and matched
                    # grounding rows.
                    "Transition geometry and supply",
                    [
                        "player_transition_edges_mean",
                        "player_transition_edges_p90",
                        "player_transition_reveal_frac",
                        "player_transition_rows_frac",
                        "player_transition_ground_rows_frac",
                    ],
                ),
                lp(
                    # THE gaming instrument. The normaliser is learnable
                    # (the state linears train under every loss, the EMA
                    # copies them), so a normalised MSE can SHRINK the
                    # unpredictable directions instead of predicting them.
                    # hp_share = the public delta's energy in the hp input
                    # subspace of public_persistent_linear; falling while
                    # gain_public rises = hp made small, not predicted (the
                    # abort -> whiten the delta).
                    "Transition: hp share of the public delta",
                    ["player_transition_hp_share"],
                ),
                lp(
                    # Column-block rms of the three state kernels (public
                    # persistent / transient, private) by feature group,
                    # against "other" (level, gender, item effect, pp, tera,
                    # volatiles, stats). hp falling relative to other is the
                    # same gaming shape read on the parameter.
                    "State kernel rms by feature group",
                    [
                        "player_state_kernel_rms_hp",
                        "player_state_kernel_rms_status",
                        "player_state_kernel_rms_boosts",
                        "player_state_kernel_rms_other",
                    ],
                ),
                lp(
                    # Drift against init: dynamics_out_proj is the ONE
                    # zero factor (exactly 0 at init; must leave it within
                    # ~200 steps or everything behind it never trains);
                    # the action table and the slot embedding start at
                    # 0.0625, chance_token_proj ~0.0625 lecun, code_table
                    # 0.088. pred_rms = rms(imagined rows) / rms(real
                    # rows) over the rows valid at t -- the off-manifold
                    # magnitude diagnostic alongside the restored consistency
                    # loss. A ratio near 1 alone does not establish alignment.
                    "Transition model: drift",
                    [
                        "player_transition_out_proj_rms",
                        "player_transition_action_table_rms",
                        "player_transition_slot_embedding_rms",
                        "player_transition_chance_token_proj_rms",
                        "player_transition_code_table_rms",
                        "player_transition_pred_rms",
                    ],
                ),
                lp(
                    # Pre-clip grad norms of the model and its parts beside
                    # the value head's TOTAL gradient (real-row CE + the
                    # imagined-row CEs under value_trains_v_head): the aux
                    # term dwarfing the value head's is it stealing the
                    # trunk.
                    "Transition model: gradient",
                    [
                        "player_transition_grad_norm",
                        "player_transition_blocks_grad_norm",
                        "player_transition_prior_grad_norm",
                        "player_transition_posterior_grad_norm",
                        "player_transition_action_encoder_grad_norm",
                        "player_transition_generator_grad_norm",
                        "player_transition_terminal_head_grad_norm",
                        "player_value_head_grad_norm",
                    ],
                ),
            ],
        ),
        ws.Section(
            # The eval slate: the policy as the training actors sample it
            # against the same policy thresholded at sampling, both EMA at
            # T=1. The search eval actor was deleted 2026-09-09 (LESSONS.md
            # "Removal ledger — 2026-09-09 search eval actor").
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
                        "player_action_head_gradient_norm",
                        "player_v_head_gradient_norm",
                    ],
                    log_y=True,
                ),
                lp(
                    "Gradient / param norm (aggregate)",
                    ["player_gradient_norm", "player_param_norm"],
                ),
                lp(
                    # What Adam APPLIED to the readout leaves a support
                    # force acts on (post-clip, post-revert rms), beside the
                    # switch_bias delta that started the pattern.
                    "Applied update rms · action readout leaves",
                    [
                        "player_applied_delta_rms_switch_bias",
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
            # Fed by the ActorStats sink (rl/environment/actor_stats.py):
            # every training actor, its env and the InferenceServer record
            # wall-time per phase, drained every actor_stats_log_steps as
            # means over the pool. Where an actor's step goes — the
            # system rate is actor-bound (learner alone ~3x faster), and
            # this is the baseline the history-carry pass is judged on.
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
                    # the suffix a request actually sends (ex.bin ~3 / ~5
                    # per request against the 64/128+ a full window pads
                    # to).
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
