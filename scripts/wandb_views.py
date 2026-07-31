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


def lp(title, y, x=None, regex=None, smooth=0.9, log_y=False):
    """Line plot with time-weighted EMA smoothing by default (smooth=0
    disables it — use for counters, where smoothing only misleads)."""
    kwargs = dict(title=title, y=y or [], log_y=log_y or None)
    if x:
        kwargs["x"] = x
    if regex:
        kwargs["metric_regex"] = regex
    if smooth:
        kwargs["smoothing_factor"] = smooth
        kwargs["smoothing_type"] = "exponentialTimeWeighted"
        kwargs["smoothing_show_original"] = True
    return wr.LinePlot(**{k: v for k, v in kwargs.items() if v is not None})


SH = "EvalActor-simpleheuristic"
# Pre-eval-rework runs: eval actor id 2 played the simple heuristic.
LEGACY = "EvalActor-full-2"


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
                    "Legacy runs (pre-rework) payoff",
                    [f"ema-payoff-{LEGACY}", f"main-payoff-{LEGACY}"],
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
            name="1 · Φ-channel health",
            is_open=True,
            panels=[
                lp("Potential adv coef (cap = 100)", ["player_potential_adv_coef"]),
                lp(
                    "Adv share: realised vs target",
                    [
                        "player_potential_adv_share",
                        "player_potential_target_adv_share",
                    ],
                ),
                lp(
                    "Gate & ensemble disagreement",
                    ["potential_gate_mean", "potential_ensemble_std_mean"],
                ),
                lp(
                    "Channel loudness",
                    [
                        "potential_phi_abs_mean",
                        "potential_phi_step_delta_abs",
                        "potential_decision_step_delta_abs",
                        "potential_dice_step_delta_abs",
                    ],
                ),
                lp(
                    "Channel quality",
                    [
                        "potential_terminal_agreement",
                        "potential_decision_share",
                        "player_potential_win_adv_corr",
                        "player_potential_adv_sign_flip",
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
                    "Gradient / param norm",
                    ["player_gradient_norm", "player_param_norm"],
                ),
                lp(
                    "Entropy",
                    [
                        "player_action_entropy",
                        "player_action_normalized_entropy",
                        "player_normalized_modality_entropy",
                    ],
                ),
                lp("Value head R²", ["player_value_head_r2"]),
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
                    # Divergence between these (init 1.0) is the readout of
                    # the per-modality logit-scale-separation hypothesis.
                    "Per-modality micro-logit scales",
                    [
                        "pi_head_modality_scale_move",
                        "pi_head_modality_scale_switch",
                        "pi_head_modality_scale_wildcard",
                        "pi_head_modality_scale_other",
                        "pi_head_modality_scale_unspecified",
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
                        "player_loss_kl",
                        "player_loss_magnet_kl",
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
                    "State advantage",
                    ["player_state_adv_mean", "player_state_adv_std"],
                ),
                lp(
                    "Channel stds",
                    ["player_win_adv_std", "player_potential_adv_std"],
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
                    "Module grad norms",
                    [
                        "player_encoder_gradient_norm",
                        "player_history_encoder_gradient_norm",
                        "player_macro_head_gradient_norm",
                        "player_pi_head_gradient_norm",
                        "player_v_head_gradient_norm",
                    ],
                    log_y=True,
                ),
            ],
        ),
        ws.Section(
            name="Throughput",
            panels=[
                lp("Frame counts", ["player_frame_count", "builder_frame_count"], smooth=0),
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


def save_view(entity, project, name, sections, update_url, settings=None):
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
        settings=ws.WorkspaceSettings(x_axis="training_step"),
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