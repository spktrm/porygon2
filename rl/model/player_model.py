from dotenv import load_dotenv

load_dotenv()
import functools
from pprint import pprint

import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from rl.environment.data import (
    CELL_MODALITY_MASK,
    NUM_MODALITY_FEATURES,
    NUM_SWITCH_CELLS,
)
from rl.environment.interfaces import (
    HistoryCarry,
    PlayerActorInput,
    PlayerActorOutput,
    PlayerEnvOutput,
    PlayerPolicyHeadOutput,
    PolicyHeadOutput,
)
from rl.environment.utils import get_ex_player_step
from rl.model.config import get_player_model_config
from rl.model.constants import (
    CLS_ROW,
    MOVE_ROWS,
    OPP_PRIVATE_ROWS,
    PRIVATE_ROWS,
    PUBLIC_ROWS,
    TARGET_ROWS,
    VALUE_CLS_ROW,
)
from rl.model.encoder import Encoder, PairValueInputs
from rl.model.heads import (
    CategoricalValueLogitHead,
    FlatActionReadout,
    HeadParams,
    PairValueHead,
    RegressionValueLogitHead,
    SlotConditioning,
    compute_policy_metrics,
    sample_categorical,
)
from rl.model.trunk import row_homogeneity
from rl.model.utils import get_num_params, prune_log_policy


def actor_params_view(variables):
    """Project checkpoint variables onto the actor's required modules.

    Historical league snapshots carry different unused learner branches.
    Strip those before JIT dispatch, retaining the complete encoder (including
    setup-time parameters) and the value output.
    Indexing required modules deliberately fails on incompatible snapshots.
    """
    required = ["encoder", "action_head", "v_head"]
    # The optional doubles readout is an actor consumer too.
    if "slot_conditioning" in variables["params"]:
        required.append("slot_conditioning")
    return {"params": {name: variables["params"][name] for name in required}}


class Porygon2PlayerModel(nn.Module):
    cfg: ConfigDict

    def setup(self):
        """Three modules: the trunk, the action readout, the critic.

        Was four, two of them the same class -- a policy ActionScoreHead and
        an advantage one over the same grid. The advantage head, `compose_q`
        and the Retrace baseline it fed retired on 2026-08-29: the policy had
        not read it since the NashPG switch, so it was a matched-control
        observer for an architecture that no longer exists, and its last
        readings are banked in the ledger.

        The action readout scores the block cells from the rows it owns;
        `v_head` reads the CLS row and nothing else.
        """
        self.encoder = Encoder(self.cfg.encoder)
        self.action_head = FlatActionReadout(self.cfg.action_head, name="action_head")
        self.v_head = CategoricalValueLogitHead(self.cfg.v_head)
        # The privileged critic (2026-09-01): architecturally identical to
        # v_head, reading VALUE_CLS -- the one row that attends over the
        # opponent-truth partition. Called only under cfg.train, so its
        # params exist in the learner-initialised tree and an actor apply
        # never visits them; nothing at deploy consumes its output.
        self.priv_v_head = CategoricalValueLogitHead(self.cfg.priv_v_head)
        # The PBRS potential channel's value (2026-09-11): learner-only, and
        # absent unless the channel runs, so strength 0 keeps today's tree.
        if self.cfg.potential_head.enabled:
            self.potential_head = RegressionValueLogitHead(self.cfg.potential_head)
        # The pairwise entity critics (2026-09-12, heads.PairValueHead):
        # learner-only, absent unless the learner's coefficient is > 0. One
        # class, two inputs -- the post-trunk public rows, and both players'
        # pre-trunk sheet latents (encoder.PairValueInputs).
        if self.cfg.pair_value_head.enabled:
            self.pair_value_public = PairValueHead(self.cfg.pair_value_head)
            self.pair_value_private = PairValueHead(self.cfg.pair_value_head)
        if self.cfg.num_decision_slots == 2:
            # Doubles only: params appear in the tree only when the module
            # is called, so singles checkpoints are unaffected.
            self.slot_conditioning = SlotConditioning()

    def _modality_log_marginal(self, log_policy: jax.Array, flat_valid_mask: jax.Array):
        """log of the policy's modality marginal over legal cells, plus each
        modality's legal-cell count — recovered by marginalisation over the
        per-cell modality constant so neither the head signature nor the
        actor payload has to carry the full distribution."""
        modality_oh = jax.nn.one_hot(
            jnp.asarray(CELL_MODALITY_MASK),
            NUM_MODALITY_FEATURES,
            dtype=log_policy.dtype,
        )
        valid_modality_mask = flat_valid_mask[..., None] * modality_oh
        modality_log_probs = nn.logsumexp(
            jnp.where(
                valid_modality_mask,
                log_policy[..., None],
                -1e9,
            ),
            axis=0,
        )
        return modality_log_probs, valid_modality_mask.sum(axis=0)

    def _calculate_entropy_metrics(
        self, modality_log_probs: jax.Array, valid_actions_per_modality: jax.Array
    ):
        modality_probs = jnp.exp(modality_log_probs)
        num_valid_modalities = (valid_actions_per_modality > 0).sum(
            dtype=modality_probs.dtype
        )
        raw_modality_entropy = -jnp.sum(
            jnp.where(
                valid_actions_per_modality > 0, modality_probs * modality_log_probs, 0.0
            )
        )
        max_modality_entropy = jnp.log(jnp.maximum(num_valid_modalities, 1.0))
        # Never 0.0, so the divide below is safe on the one-live-modality row
        # that the outer jnp.where discards anyway.
        safe_max_modality_entropy = jnp.where(
            num_valid_modalities > 1, max_modality_entropy, 1.0
        )
        return jnp.where(
            num_valid_modalities > 1,
            raw_modality_entropy / safe_max_modality_entropy,
            0.0,
        )

    def _forward_action_head(
        self,
        sequence_rows: tuple[jax.Array, jax.Array, jax.Array],
        valid_mask: jax.Array,
        head: PolicyHeadOutput,
        train: bool,
        temp: float,
        prune_threshold: float = 0.0,
    ):
        """Dispatch on decision slots: singles = one flat categorical over
        the block cells (the historical path, unchanged); doubles = two head-level
        stages over per-slot masks with slot 2 conditioned on slot 1's
        choice — the trunk is forwarded once either way."""
        if self.cfg.num_decision_slots == 2:
            return self._forward_two_slots(
                sequence_rows, valid_mask, head, train, temp, prune_threshold
            )
        return self._forward_single_slot(
            sequence_rows, valid_mask, head, train, temp, prune_threshold
        )

    def _legal_logits(
        self,
        sequence_rows: tuple[jax.Array, jax.Array, jax.Array],
        valid_mask: jax.Array,
        temp: float,
        decision_slot: int = 0,
    ) -> jax.Array:
        """The readout's logits with illegal cells at -1e9 (finite, so no
        `-inf * 0` in a vjp): the one form the sampler reads.
        `decision_slot` picks the ally row the switch
        block reads -- 0 in singles, 1 for doubles stage 2."""
        private_rows, move_rows, target_rows = sequence_rows
        logits = self.action_head(
            private_rows, move_rows, target_rows, temp=temp, decision_slot=decision_slot
        )
        return jnp.where(valid_mask, logits, -1e9)

    def _score_and_sample(
        self,
        sequence_rows: tuple[jax.Array, jax.Array, jax.Array],
        valid_mask: jax.Array,
        given_index: jax.Array | None,
        temp: float,
        prune_threshold: float = 0.0,
        decision_slot: int = 0,
    ):
        """Score one decision's cells and pick an action.

        THE policy scoring path — singles calls it once, doubles calls it
        once per stage with shared params. `given_index` teacher-forces the
        stored choice so the learner's recompute conditions on what the
        actor actually did; None samples.

        Behaviour policy mu == pi, with illegal cells at the dtype's min so
        the sampler can never draw one. `prune_threshold` removes the
        legal cells below it from mu ONLY (rl/model/utils.py
        prune_log_policy; 0.0 is bit-identical): the metrics read pi
        untouched, and the stored log_prob is mu's, so an eval slot
        sampling the thresholded policy reports what it sampled.
        """
        flat_valid = valid_mask
        pi_logits = self._legal_logits(sequence_rows, valid_mask, temp, decision_slot)
        # prior=None is uniform over legal cells -- which is exactly what the
        # flat readout's all-zero init produces, so the init policy and the
        # metric anchor are the same distribution.
        metrics = compute_policy_metrics(logits=pi_logits, valid_mask=flat_valid)
        log_mu = prune_log_policy(metrics.log_policy, flat_valid, prune_threshold)
        if given_index is not None:
            action_index = given_index
        else:
            action_index = sample_categorical(log_mu, self.make_rng("sampling"))
        log_prob = jnp.take(log_mu, action_index, axis=-1)
        return flat_valid, metrics, action_index, log_prob

    def _forward_single_slot(
        self,
        sequence_rows: tuple[jax.Array, jax.Array, jax.Array],
        valid_mask: jax.Array,
        head: PolicyHeadOutput,
        train: bool,
        temp: float,
        prune_threshold: float = 0.0,
    ):
        if train:
            given_index = head.action_index
        else:
            given_index = None
        flat_valid, metrics, action_index, log_prob = self._score_and_sample(
            sequence_rows,
            valid_mask,
            given_index,
            temp,
            prune_threshold,
        )
        learner_only = {}
        if self.cfg.train:
            learner_only = {
                "log_policy": metrics.log_policy,
            }
        modality_log_probs, valid_per_modality = self._modality_log_marginal(
            metrics.log_policy, flat_valid
        )
        return PlayerPolicyHeadOutput(
            action_index=action_index,
            log_prob=log_prob,
            **learner_only,
            entropy=metrics.entropy,
            normalized_entropy=metrics.normalized_entropy,
            magnet_kl=metrics.magnet_kl,
            normalized_modality_entropy=self._calculate_entropy_metrics(
                modality_log_probs, valid_per_modality
            ),
        )

    def _apply_choice_collision(self, valid_mask: jax.Array, action_index: jax.Array):
        """Slot-2 legality given slot 1's choice: both mons cannot switch
        to the same reserve. A switch cell IS its reserve index in the block
        space, so the collision is exactly slot 1's own cell. Must be applied
        identically at act and learn time or the stored behaviour log-prob
        and the learner's recompute diverge."""
        a1_is_switch = action_index < NUM_SWITCH_CELLS
        collide = a1_is_switch & (jnp.arange(valid_mask.shape[-1]) == action_index)
        return jnp.where(collide, False, valid_mask)

    def _forward_two_slots(
        self,
        sequence_rows: tuple[jax.Array, jax.Array, jax.Array],
        valid_mask: jax.Array,
        head: PolicyHeadOutput,
        train: bool,
        temp: float,
        prune_threshold: float = 0.0,
    ):
        """Doubles: valid_mask is (2, NUM_ACTION_CELLS) per-slot masks and, in train,
        head.action_index is (2,). One trunk pass serves both decisions —
        only the heads run twice, with slot 2's embeddings conditioned on
        slot 1's chosen action and its mask adjusted for choice collisions.
        Joint log_prob is the stage sum (chain rule) — exact, so
        v-trace/SPO consume it unchanged. entropy and magnet_kl are
        single-sample estimators of the joint quantities (stage-1 term
        plus the stage-2 term at the realised a1): unbiased at act time
        where a1 ~ pi_1, teacher-forced at learn time where a1 comes from
        the behaviour policy — a documented, standard off-policy bias; the
        magnet gradient also drops the REINFORCE pathway through pi_1
        reweighting the conditional KLs. NOTE: the service/actor/replay
        plumbing for this path (per-slot masks in requests, two stored
        action indices, (2, NUM_ACTION_CELLS) full-support log_policy in the learner)
        is the remaining doubles workstream; the model side is complete.
        """
        if train:
            stage1_given = head.action_index[0]
        else:
            stage1_given = None
        flat_valid_1, metrics_1, index_1, log_prob_1 = self._score_and_sample(
            sequence_rows, valid_mask[0], stage1_given, temp, None, prune_threshold
        )

        cond_rows = self.slot_conditioning(sequence_rows, index_1)
        mask_2 = self._apply_choice_collision(valid_mask[1], index_1)
        if train:
            stage2_given = head.action_index[1]
        else:
            stage2_given = None
        flat_valid_2, metrics_2, index_2, log_prob_2 = self._score_and_sample(
            cond_rows,
            mask_2,
            stage2_given,
            temp,
            None,
            prune_threshold,
            decision_slot=1,
        )

        action_index = jnp.stack([index_1, index_2])
        # Diagnostic average of the per-stage values (a true joint version
        # would need raw/max modality entropies threaded out; not worth it
        # for telemetry).
        normalized_modality_entropy = (
            self._calculate_entropy_metrics(
                *self._modality_log_marginal(metrics_1.log_policy, flat_valid_1)
            )
            + self._calculate_entropy_metrics(
                *self._modality_log_marginal(metrics_2.log_policy, flat_valid_2)
            )
        ) / 2.0

        # Joint normalised entropy: (H1 + H2) / (log N1 + log N2) — the
        # stage with the bigger branching factor carries proportionally
        # more of the normaliser (a mean of per-stage ratios would weight
        # a 2-option stage equally with a 20-option one). Forced stages
        # (N <= 1, H = 0) drop out of numerator and denominator alike.
        entropy = metrics_1.entropy + metrics_2.entropy
        num_valid_1 = flat_valid_1.sum()
        num_valid_2 = flat_valid_2.sum()
        denom = jnp.where(
            num_valid_1 > 1, jnp.log(jnp.maximum(num_valid_1, 2)), 0.0
        ) + jnp.where(num_valid_2 > 1, jnp.log(jnp.maximum(num_valid_2, 2)), 0.0)
        denom = denom.astype(entropy.dtype)
        normalized_entropy = jnp.where(
            denom > 0, entropy / jnp.maximum(denom, 1e-9), 0.0
        )

        if self.cfg.train:
            log_policy = jnp.stack([metrics_1.log_policy, metrics_2.log_policy])
        else:
            log_policy = ()
        return PlayerPolicyHeadOutput(
            action_index=action_index,
            log_prob=log_prob_1 + log_prob_2,
            log_policy=log_policy,
            entropy=entropy,
            normalized_entropy=normalized_entropy,
            magnet_kl=metrics_1.magnet_kl + metrics_2.magnet_kl,
            normalized_modality_entropy=normalized_modality_entropy,
        )

    def get_head_outputs(
        self,
        sequence: jax.Array,
        row_valid: jax.Array,
        trunk_out_group_l2: tuple[jax.Array, jax.Array] | None,
        pair_value_inputs: PairValueInputs | tuple,
        env_step: PlayerEnvOutput,
        actor_output: PlayerActorOutput,
        head_params: HeadParams,
        history_stats: dict[str, jax.Array],
        history_carry: HistoryCarry,
    ):
        """Each head slices the rows it owns, by name. No head ever carries a
        row offset -- rl/model/constants.py derives them once.

        history_stats and history_carry are per TRAJECTORY (the history is
        shared across the requests); closed over rather than mapped, so the
        vmap in __call__ broadcasts them to one copy per step."""
        action_head = self._forward_action_head(
            (sequence[PRIVATE_ROWS], sequence[MOVE_ROWS], sequence[TARGET_ROWS]),
            env_step.action_mask,
            actor_output.action_head,
            train=self.cfg.train,
            temp=head_params.temp,
            prune_threshold=head_params.prune_threshold,
        )
        learner_only = {}
        if self.cfg.train:
            # Rows converging to one direction reads on the existing panels
            # as "entropy at ceiling while the pointer params grow" -- the
            # phase-1 support-anchor shape -- so it gets its own reading.
            # Offline twin: rl/offline/trunk_homogeneity.py, per block.
            row_cosine, row_participation = row_homogeneity(sequence)
            group_l2_sum, group_rows = trunk_out_group_l2
            learner_only = {
                # The privileged critic: VALUE_CLS, and only VALUE_CLS.
                "priv_value_head": self.priv_v_head(sequence[VALUE_CLS_ROW]),
                "trunk_row_cosine": row_cosine,
                "trunk_row_participation": row_participation,
                "trunk_out_group_l2_sum": group_l2_sum,
                "trunk_out_group_rows": group_rows,
                # The History panels: the step GAT's read and the
                # backbone's write gate (history_encoder.history_step_stats).
                "history_step_attn_entropy": history_stats["step_attn_entropy"],
                "history_step_attn_to_src": history_stats["step_attn_to_src"],
                "history_step_attn_to_src_uniform": history_stats[
                    "step_attn_to_src_uniform"
                ],
                "history_gate_mean": history_stats["gate_mean"],
            }
            if self.cfg.potential_head.enabled:
                # CLS under stop_gradient: the channel's label is the
                # human-fitted potential, so its loss reaches this head and
                # nothing it reads (test_potential_slot, the reach test).
                learner_only["potential_head"] = self.potential_head(
                    jax.lax.stop_gradient(sequence[CLS_ROW])
                )
            if self.cfg.pair_value_head.enabled:
                # Both heads read POST-trunk rows (user call 2026-09-12: the
                # trunk routes whatever context a row needs; no field
                # context of the head's own). Public: the public rows, mine
                # first by layout. Private: my sheet rows then the
                # opponent-truth rows, learner-only by the read mask.
                # Gradient live into the trunk through every row each head
                # reads (the CLS critics are the matched control).
                learner_only["pair_value_public"] = self.pair_value_public(
                    sequence[PUBLIC_ROWS],
                    row_valid[PUBLIC_ROWS],
                    pair_value_inputs.public_alive,
                )
                learner_only["pair_value_private"] = self.pair_value_private(
                    jnp.concatenate(
                        (sequence[PRIVATE_ROWS], sequence[OPP_PRIVATE_ROWS]), axis=0
                    ),
                    jnp.concatenate(
                        (row_valid[PRIVATE_ROWS], row_valid[OPP_PRIVATE_ROWS])
                    ),
                    pair_value_inputs.sheet_alive,
                )
        return PlayerActorOutput(
            action_head=action_head,
            # The CLS row, and only the CLS row.
            value_head=self.v_head(sequence[CLS_ROW]),
            history_carry=history_carry,
            **learner_only,
        )

    def __call__(
        self,
        actor_input: PlayerActorInput,
        actor_output: PlayerActorOutput,
        head_params: HeadParams,
    ):
        """
        Shared forward pass for encoder and policy head.
        """
        (
            sequence,
            row_valid,
            trunk_out_group_l2,
            pair_value_inputs,
            history_stats,
            history_carry,
        ) = self.encoder(
            actor_input.env,
            actor_input.packed_history,
            actor_input.history,
            actor_input.history_carry,
        )

        output = jax.vmap(
            functools.partial(
                self.get_head_outputs,
                head_params=head_params,
                history_stats=history_stats,
                history_carry=history_carry,
            )
        )(
            sequence,
            row_valid,
            trunk_out_group_l2,
            pair_value_inputs,
            actor_input.env,
            actor_output,
        )
        return output


def get_player_model(config: ConfigDict = None) -> nn.Module:
    if config is None:
        config = get_player_model_config()
    return Porygon2PlayerModel(config)


def main(generation: int = 9):
    """Init the learner network on an example step and print param counts.

    Attention-map dumps live in scripts/attn_probe.py.
    """
    learner_network = get_player_model(get_player_model_config(generation, train=True))

    ex_actor_input, ex_actor_output = jax.device_put(
        jax.tree.map(lambda x: x[:, 0], get_ex_player_step())
    )
    key = jax.random.key(42)
    params = learner_network.init(key, ex_actor_input, ex_actor_output, HeadParams())
    pprint(get_num_params(params), sort_dicts=False)


if __name__ == "__main__":
    main()
