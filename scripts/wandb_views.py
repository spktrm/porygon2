"""Build the saved wandb workspace views for this project's dashboards.

Creates/refreshes two views:
  - pokemon-rl         -> "Signal health"  (training-run diagnostics)
  - pokemon-rl-offline -> "Critic health"  (offline critic / Phi ensemble)

Panel keys mirror what rl/main.py and rl/offline/train.py log; when metrics
are added or renamed, update the sections here and re-run. Note each run
SAVES A NEW VIEW (the API matches by internal id, not display name), so
delete the superseded view in the wandb UI afterwards — or pass the old
view's URL via --update-url to edit it in place.

Usage:
    python scripts/wandb_views.py [--entity ENTITY]
        [--update-rl-url URL] [--update-offline-url URL]

Requires `pip install wandb-workspaces` and a logged-in wandb credential.
"""

import argparse

import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws


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
                    x="training_step",
                    smooth=0,
                ),
                lp(
                    "Smoothed margin (alive-mon diff)",
                    [f"smoothed-margin-{SH}-{i}" for i in range(3)],
                    x="training_step",
                    smooth=0,
                ),
                lp(
                    # Payoff (-1/0/+1), not the wr keys: runs before Aug 2026
                    # logged wr as booleans, which the wandb UI renders as
                    # NaN. Smoothed payoff reads as 2*winrate - 1.
                    "Raw payoff per actor (UI-smoothed)",
                    [f"ema-payoff-{SH}-{i}" for i in range(3)],
                    x="training_step",
                    smooth=0.95,
                ),
                lp(
                    "Main-params sanity check",
                    [f"main-payoff-{SH}-{i}" for i in range(3)]
                    + [f"main-margin-{SH}-{i}" for i in range(3)],
                    x="training_step",
                    smooth=0.9,
                ),
                lp(
                    "Eval games played",
                    [f"games-{SH}-{i}" for i in range(3)],
                    x="training_step",
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
                    # multi-lambda aux CE (representation shaping only).
                    "Value losses (main + multi-lambda aux)",
                    ["player_loss_v_win", "player_loss_v_aux"],
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
                    # gap is honest and the fix is opponent pressure
                    # (stage 4), not the improvement term.
                    "Empirical returns: voluntary switch vs move",
                    [
                        "player_q_target_voluntary_switch",
                        "player_q_target_move",
                    ],
                ),
                lp(
                    # Stage 4 — cross-population intake. foreign_frac is
                    # the realised share of Q training data drawn from
                    # exploiter buffers (0 until the exploiters exist);
                    # r2_foreign persistently below player_q_r2 means the
                    # intake is too off-policy to learn from (Retrace
                    # cutting every trace) rather than free switching
                    # counterfactuals.
                    "Cross-population Q intake",
                    ["player_q_foreign_frac", "player_q_r2_foreign"],
                ),
                lp(
                    "Q loss & head gradient",
                    ["player_loss_q", "player_q_head_gradient_norm"],
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
                wr.MediaBrowser(
                    title="League win-rate heatmap (full pairwise matrix)",
                    media_keys=["league_winrate_heatmap"],
                    num_columns=1,
                    gallery_axis="step",
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
                    "Trajectory lengths",
                    [
                        "player_trajectory_length_mean",
                        "player_trajectory_length_min",
                        "player_trajectory_length_max",
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
                lp("Training step", ["training_step"], smooth=0),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="jtwin")
    parser.add_argument("--update-rl-url", default=None)
    parser.add_argument("--update-offline-url", default=None)
    args = parser.parse_args()

    save_view(
        args.entity,
        "pokemon-rl",
        "Signal health",
        rl_sections(),
        args.update_rl_url,
        # Learner-step x-axis for every panel: all learner and eval log rows
        # carry training_step (rl/learner/learner.py, rl/main.py), and it is
        # comparable across runs unlike wandb's _step row counter. The
        # offline project does not log this key, so it keeps the default.
        # force_x pins it per panel — the workspace-level setting alone is
        # overridden by the "Step" default materialised onto each panel.
        settings=ws.WorkspaceSettings(x_axis="training_step"),
        force_x="training_step",
    )
    save_view(
        args.entity,
        "pokemon-rl-offline",
        "Critic health",
        offline_sections(),
        args.update_offline_url,
    )


if __name__ == "__main__":
    main()
