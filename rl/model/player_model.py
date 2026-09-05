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
    NUM_SPECIES,
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
from rl.environment.protos.features_pb2 import (
    EntityPrivateNodeFeature,
    EntityRevealedNodeFeature,
    InfoFeature,
)
from rl.environment.utils import get_ex_player_step
from rl.model.config import get_player_model_config
from rl.model.constants import (
    CLS_ROW,
    DYNAMICS_GROUP_SLICES,
    MOVE_ROWS,
    NUM_PRIVATE_SLOTS,
    NUM_PUBLIC_SLOTS,
    POLICY_READABLE_ROWS,
    PRIVATE_ROWS,
    PUBLIC_ROWS,
    TARGET_ROWS,
    VALUE_CLS_ROW,
)
from rl.model.encoder import belief_alignment  # noqa: F401  (tests import it here)
from rl.model.encoder import Encoder, OppCodeLabels
from rl.model.heads import (
    CategoricalValueLogitHead,
    FlatActionReadout,
    HeadParams,
    SlotConditioning,
    compute_policy_metrics,
    sample_categorical,
)
from rl.model.modules import MLP
from rl.model.transition import TransitionModel
from rl.model.trunk import row_homogeneity
from rl.model.utils import get_num_params, legal_log_policy


def _sampling_log_policy(log_policy: jax.Array, valid_mask: jax.Array) -> jax.Array:
    """log pi with illegal cells at the dtype's min, for sample_categorical."""
    return jnp.where(valid_mask, log_policy, jnp.finfo(log_policy.dtype).min)


def _match_rows(key_now: jax.Array, key_next: jax.Array, valid_now, valid_next):
    """Row i now -> the row carrying the same key next step. (matched, index)."""
    hits = (
        (key_now[:, None] == key_next[None, :])
        & valid_now[:, None]
        & valid_next[None, :]
    )
    return hits.any(axis=-1), jnp.argmax(hits, axis=-1)


def dynamics_alignment(env_now: PlayerEnvOutput, env_next: PlayerEnvOutput):
    """Which next-step target row is this step's target row j? (2026-09-03)

    The dynamics head's rows are DYNAMICS_TARGET_ROWS: public 12, my
    private 6, field 3. Public rows re-sort every step (actives first), so
    row i now and row i next are different mons after a switch; PUBLIC_ORDER
    carries each row's stable entity index and the match is a (12, 12)
    equality over it -- `belief_alignment`'s construction over TIME instead
    of over sides. The private rows follow the request's own order, which
    also moves on a switch, so they match on ENTITY_IDX (1 + stable index;
    0 = never fielded, unmatched and skipped -- its row is static anyway).
    Field rows are fixed slots. Returns (matched (21,), next_index (21,)),
    the index valid only where matched.
    """
    order_slice = slice(
        InfoFeature.INFO_FEATURE__PUBLIC_ORDER_0,
        InfoFeature.INFO_FEATURE__PUBLIC_ORDER_11 + 1,
    )
    order_now = env_now.info[order_slice]
    order_next = env_next.info[order_slice]
    order_ok_now = (order_now >= 0) & (order_now < NUM_PUBLIC_SLOTS)
    order_ok_next = (order_next >= 0) & (order_next < NUM_PUBLIC_SLOTS)
    public_matched, public_index = _match_rows(
        order_now, order_next, order_ok_now, order_ok_next
    )

    idx_column = EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__ENTITY_IDX
    idx_now = env_now.private_team[:, idx_column]
    idx_next = env_next.private_team[:, idx_column]
    private_matched, private_index = _match_rows(
        idx_now, idx_next, idx_now > 0, idx_next > 0
    )

    field = DYNAMICS_GROUP_SLICES["field"]
    num_field = field.stop - field.start
    field_index = jnp.arange(num_field, dtype=public_index.dtype)

    matched = jnp.concatenate(
        (public_matched, private_matched, jnp.ones(num_field, dtype=jnp.bool_))
    )
    next_index = jnp.concatenate(
        (
            public_index,
            NUM_PUBLIC_SLOTS + private_index,
            NUM_PUBLIC_SLOTS + NUM_PRIVATE_SLOTS + field_index,
        )
    )
    return matched, next_index


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
        # The belief head (2026-09-01): from each opponent mon's PUBLIC row
        # (post-trunk, policy-readable -- deduction from what the agent can
        # see), predict that mon's discrete code. CE against the sg'd code
        # is belief-state shaping: the act-time information set is
        # untouched (the partition test pins it), the gradient just asks
        # public representations to be DECODABLE to the truth.
        self.belief_head = MLP(**self.cfg.belief_head.mlp.to_dict())
        # The species-only matched control (2026-09-02): the same (G, K)
        # logits from NOTHING but the matched public row's species token,
        # a table lookup. The belief head reads a post-trunk row that
        # already carries the species (one shared species embedder feeds
        # both rows), so `player_belief_accuracy` alone cannot tell "this
        # species' typical code" from a belief formed from what the
        # opponent has shown. Its input is an integer: no shared param
        # ever receives its gradient, and its CE enters the loss unscaled
        # because a coefficient could only ever be 1.0.
        code = self.cfg.encoder.opp_code
        self.species_belief = nn.Embed(
            NUM_SPECIES,
            code.num_groups * code.num_classes,
            dtype=self.cfg.dtype,
            name="species_belief",
        )
        # The revealed-row matched control (2026-09-04): the same (G, K)
        # logits from the matched mon's own PRE-trunk public row and
        # nothing else -- species, revealed moves, item, ability, learnset,
        # state, active, exactly what that row says in isolation. The
        # species table brackets the belief head from below by species
        # only; this one brackets it by everything the mon has SHOWN, so
        # `player_belief_context_margin` (belief minus this) is the part
        # of the belief formed from context: history, the other rows, what
        # the opponent has and has not done. Input under stop_gradient: no
        # shared param receives its gradient; its CE enters unscaled.
        self.revealed_belief = MLP(**self.cfg.revealed_belief.mlp.to_dict())
        # The latent transition model (2026-09-05, rl/model/transition.py):
        # g(h_t, a, z) over the post-trunk policy-readable rows. Its params
        # are lazy, so instantiating it here costs the actor nothing; it is
        # CALLED only under cfg.train (Step 3's search will call `imagine`).
        self.transition = TransitionModel(
            self.cfg.transition, dtype=self.cfg.dtype, name="transition"
        )
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
    ):
        """Dispatch on decision slots: singles = one flat categorical over
        the block cells (the historical path, unchanged); doubles = two head-level
        stages over per-slot masks with slot 2 conditioned on slot 1's
        choice — the trunk is forwarded once either way."""
        if self.cfg.num_decision_slots == 2:
            return self._forward_two_slots(sequence_rows, valid_mask, head, train, temp)
        return self._forward_single_slot(sequence_rows, valid_mask, head, train, temp)

    def _score_and_sample(
        self,
        sequence_rows: tuple[jax.Array, jax.Array, jax.Array],
        valid_mask: jax.Array,
        given_index: jax.Array | None,
        temp: float,
    ):
        """Score one decision's cells and pick an action.

        THE policy scoring path — singles calls it once, doubles calls it
        once per stage with shared params. `given_index` teacher-forces the
        stored choice so the learner's recompute conditions on what the
        actor actually did; None samples.

        Behaviour policy mu == pi, with illegal cells at the dtype's min so
        the sampler can never draw one.
        """
        private_rows, move_rows, target_rows = sequence_rows
        logits = self.action_head(private_rows, move_rows, target_rows, temp=temp)
        flat_valid = valid_mask
        pi_logits = jnp.where(flat_valid, logits, -1e9)
        # prior=None is uniform over legal cells -- which is exactly what the
        # flat readout's all-zero init produces, so the init policy and the
        # metric anchor are the same distribution.
        metrics = compute_policy_metrics(logits=pi_logits, valid_mask=flat_valid)
        log_mu = _sampling_log_policy(metrics.log_policy, flat_valid)
        action_index = (
            given_index
            if given_index is not None
            else sample_categorical(log_mu, self.make_rng("sampling"))
        )
        log_prob = jnp.take(log_mu, action_index, axis=-1)
        return flat_valid, metrics, action_index, log_prob

    def _forward_single_slot(
        self,
        sequence_rows: tuple[jax.Array, jax.Array, jax.Array],
        valid_mask: jax.Array,
        head: PolicyHeadOutput,
        train: bool,
        temp: float,
    ):
        flat_valid, metrics, action_index, log_prob = self._score_and_sample(
            sequence_rows, valid_mask, head.action_index if train else None, temp
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
        stage1_given = head.action_index[0] if train else None
        flat_valid_1, metrics_1, index_1, log_prob_1 = self._score_and_sample(
            sequence_rows, valid_mask[0], stage1_given, temp
        )

        cond_rows = self.slot_conditioning(sequence_rows, index_1)
        mask_2 = self._apply_choice_collision(valid_mask[1], index_1)
        stage2_given = head.action_index[1] if train else None
        flat_valid_2, metrics_2, index_2, log_prob_2 = self._score_and_sample(
            cond_rows, mask_2, stage2_given, temp
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

        return PlayerPolicyHeadOutput(
            action_index=action_index,
            log_prob=log_prob_1 + log_prob_2,
            log_policy=(
                jnp.stack([metrics_1.log_policy, metrics_2.log_policy])
                if self.cfg.train
                else ()
            ),
            entropy=entropy,
            normalized_entropy=normalized_entropy,
            magnet_kl=metrics_1.magnet_kl + metrics_2.magnet_kl,
            normalized_modality_entropy=normalized_modality_entropy,
        )

    def get_head_outputs(
        self,
        sequence: jax.Array,
        opp_code_labels: OppCodeLabels,
        dynamics_rows: jax.Array,
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
        )
        learner_only = {}
        if self.cfg.train:
            matched = opp_code_labels.matched
            public_row_index = opp_code_labels.public_row_index
            matched_rows = sequence[PUBLIC_ROWS][public_row_index]
            code_shape = opp_code_labels.code.shape
            belief_logits = self.belief_head(matched_rows).reshape(code_shape)
            # Keyed on the PUBLIC row's species, never the private one: under
            # Illusion the board shows the disguise, and a control that sees
            # more than the row it is matched against is not matched.
            public_species = env_step.revealed_team[
                public_row_index,
                EntityRevealedNodeFeature.ENTITY_REVEALED_NODE_FEATURE__SPECIES,
            ]
            species_belief_logits = self.species_belief(public_species).reshape(
                code_shape
            )
            revealed_rows = jax.lax.stop_gradient(
                dynamics_rows[DYNAMICS_GROUP_SLICES["public"]][public_row_index]
            )
            revealed_belief_logits = self.revealed_belief(revealed_rows).reshape(
                code_shape
            )
            # Rows converging to one direction reads on the existing panels
            # as "entropy at ceiling while the pointer params grow" -- the
            # phase-1 support-anchor shape -- so it gets its own reading.
            # Offline twin: rl/offline/trunk_homogeneity.py, per block.
            row_cosine, row_participation = row_homogeneity(sequence)
            learner_only = {
                # The grounding label: the target rows' pre-trunk content.
                # The transition model's grounding head is scored against
                # the NEXT step's copy of this (train_step.dynamics_losses).
                "dynamics_target": dynamics_rows,
                # The privileged critic: VALUE_CLS, and only VALUE_CLS.
                "priv_value_head": self.priv_v_head(sequence[VALUE_CLS_ROW]),
                # The critic's code (the secret rows' content) and the
                # belief head's label -- the code of the HIDDEN tokens only
                # (encoder.OppCodeLabels).
                "opp_code": opp_code_labels.code,
                "hidden_code": opp_code_labels.hidden_code,
                "belief_logits": belief_logits,
                "species_belief_logits": species_belief_logits,
                "revealed_belief_logits": revealed_belief_logits,
                "belief_matched": matched,
                "belief_hidden_any": opp_code_labels.hidden_any,
                "trunk_row_cosine": row_cosine,
                "trunk_row_participation": row_participation,
                # The History panels: the step GAT's read and the
                # backbone's write gate (history_encoder.history_step_stats).
                "history_step_attn_entropy": history_stats["step_attn_entropy"],
                "history_step_attn_to_src": history_stats["step_attn_to_src"],
                "history_step_attn_to_src_uniform": history_stats[
                    "step_attn_to_src_uniform"
                ],
                "history_gate_mean": history_stats["gate_mean"],
            }
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
            opp_code_labels,
            dynamics_rows,
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
        )(sequence, opp_code_labels, dynamics_rows, actor_input.env, actor_output)
        if self.cfg.train:
            output = output.replace(
                **self._forward_transition(
                    sequence, row_valid, output, actor_input.env.action_mask
                )
            )
        return output

    def _forward_transition(
        self,
        sequence: jax.Array,
        row_valid: jax.Array,
        output: PlayerActorOutput,
        action_mask: jax.Array,
    ) -> dict[str, jax.Array]:
        """The transition model over the trajectory (T, rows, D): every
        step is paired with its POSITIONAL successor (the last step with
        itself; train_step masks it). Only the policy-readable rows enter
        -- the rollout's information set -- and the posterior's read of
        the real next rows is under stop_gradient: the model predicts the
        trunk, it never trains it through the target. The shared value
        head and action readout are applied to the imagined rows unchanged,
        so their outputs on imagined states mean what they mean on real
        ones; the policy is masked by the REAL next legal set."""
        rows = sequence[:, POLICY_READABLE_ROWS]
        valid = row_valid[:, POLICY_READABLE_ROWS]
        next_rows = jax.lax.stop_gradient(
            jnp.concatenate((rows[1:], rows[-1:]), axis=0)
        )
        next_valid = jnp.concatenate((valid[1:], valid[-1:]), axis=0)
        next_mask = jnp.concatenate((action_mask[1:], action_mask[-1:]), axis=0)
        transition = self.transition(
            rows, valid, output.action_head.action_index, next_rows, next_valid
        )
        pred = transition.pred
        logits = jax.vmap(
            lambda imagined: self.action_head(
                imagined[PRIVATE_ROWS], imagined[MOVE_ROWS], imagined[TARGET_ROWS]
            )
        )(pred)
        error = (pred.astype(jnp.float32) - next_rows.astype(jnp.float32)) ** 2
        movement = (next_rows.astype(jnp.float32) - rows.astype(jnp.float32)) ** 2
        return {
            "transition_cons_err": error.sum(axis=-1),
            "transition_cons_scale": jax.lax.stop_gradient(movement.sum(axis=-1)),
            "transition_prior_logits": transition.prior_logits,
            "transition_post_logits": transition.post_logits,
            "transition_ground": transition.ground,
            "transition_ground_prior": transition.ground_prior,
            "transition_value_head": self.v_head(pred[:, CLS_ROW]),
            "transition_log_policy": legal_log_policy(logits, next_mask),
            "transition_mask_logits": transition.mask_logits,
            "transition_kind_logits": transition.kind_logits,
            "transition_done_logit": transition.done_logit,
        }


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
