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
    MOVE_ROWS,
    NUM_PUBLIC_SLOTS,
    PRIVATE_ROWS,
    PUBLIC_ROWS,
    TARGET_ROWS,
    VALUE_CLS_ROW,
)
from rl.model.encoder import Encoder
from rl.model.heads import (
    CategoricalValueLogitHead,
    FlatActionReadout,
    HeadParams,
    SlotConditioning,
    compute_policy_metrics,
    sample_categorical,
)
from rl.model.modules import MLP
from rl.model.trunk import row_homogeneity
from rl.model.utils import get_num_params


def _sampling_log_policy(log_policy: jax.Array, valid_mask: jax.Array) -> jax.Array:
    """log pi with illegal cells at the dtype's min, for sample_categorical."""
    return jnp.where(valid_mask, log_policy, jnp.finfo(log_policy.dtype).min)


def belief_alignment(opp_private_team: jax.Array, info: jax.Array):
    """Which public row is opponent sheet row j? (2026-09-01)

    The sheet's ENTITY_IDX (1 + stable index, 0 = never fielded) against
    PUBLIC_ORDER, restricted to the OPPONENT half of the public rows --
    a my-side mon can never legitimately match there, so a bogus index
    cannot alias across sides. Returns (matched (6,), public_row_index (6,)
    valid only where matched). A still-disguised or never-fielded mon is
    simply unmatched: the belief loss skips it.
    """
    opp_idx = opp_private_team[
        :, EntityPrivateNodeFeature.ENTITY_PRIVATE_NODE_FEATURE__ENTITY_IDX
    ]
    public_order = info[
        InfoFeature.INFO_FEATURE__PUBLIC_ORDER_0 : InfoFeature.INFO_FEATURE__PUBLIC_ORDER_11
        + 1
    ]
    opp_half = jnp.arange(NUM_PUBLIC_SLOTS) >= NUM_PUBLIC_SLOTS // 2
    hits = (
        (public_order[None, :] == (opp_idx[:, None] - 1))
        & opp_half[None, :]
        & (opp_idx[:, None] > 0)
    )
    matched = hits.any(axis=-1)
    public_row_index = jnp.argmax(hits, axis=-1)
    return matched, public_row_index


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
        opp_code_one_hot: jax.Array,
        env_step: PlayerEnvOutput,
        actor_output: PlayerActorOutput,
        head_params: HeadParams,
    ):
        """Each head slices the rows it owns, by name. No head ever carries a
        row offset -- rl/model/constants.py derives them once."""
        action_head = self._forward_action_head(
            (sequence[PRIVATE_ROWS], sequence[MOVE_ROWS], sequence[TARGET_ROWS]),
            env_step.action_mask,
            actor_output.action_head,
            train=self.cfg.train,
            temp=head_params.temp,
        )
        learner_only = {}
        if self.cfg.train:
            matched, public_row_index = belief_alignment(
                env_step.opp_private_team, env_step.info
            )
            matched_rows = sequence[PUBLIC_ROWS][public_row_index]
            code_shape = opp_code_one_hot.shape
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
            # Rows converging to one direction reads on the existing panels
            # as "entropy at ceiling while the pointer params grow" -- the
            # phase-1 support-anchor shape -- so it gets its own reading.
            # Offline twin: rl/offline/trunk_homogeneity.py, per block.
            row_cosine, row_participation = row_homogeneity(sequence)
            learner_only = {
                # The privileged critic: VALUE_CLS, and only VALUE_CLS.
                "priv_value_head": self.priv_v_head(sequence[VALUE_CLS_ROW]),
                # The belief label, straight from where the secret rows
                # were coded.
                "opp_code": opp_code_one_hot,
                "belief_logits": belief_logits,
                "species_belief_logits": species_belief_logits,
                "belief_matched": matched,
                "trunk_row_cosine": row_cosine,
                "trunk_row_participation": row_participation,
            }
        return PlayerActorOutput(
            action_head=action_head,
            # The CLS row, and only the CLS row.
            value_head=self.v_head(sequence[CLS_ROW]),
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
        sequence, opp_code_one_hot = self.encoder(
            actor_input.env, actor_input.packed_history, actor_input.history
        )

        return jax.vmap(
            functools.partial(self.get_head_outputs, head_params=head_params)
        )(sequence, opp_code_one_hot, actor_input.env, actor_output)


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
