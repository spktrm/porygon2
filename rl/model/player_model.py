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
    ConsequenceInputs,
    ConsequenceOutput,
    HistoryCarry,
    PlayerActorInput,
    PlayerActorOutput,
    PlayerEnvOutput,
    PlayerPolicyHeadOutput,
    PolicyHeadOutput,
)
from rl.environment.protos.features_pb2 import EntityPublicNodeFeature
from rl.environment.utils import get_ex_player_step
from rl.model.config import get_player_model_config
from rl.model.consequence import (
    CONSEQUENCE_NOISE_SIZE,
    CONSEQUENCE_ROWS,
    ConsequenceModel,
)
from rl.model.constants import (
    CLS_ROW,
    MOVE_ROWS,
    PRIVATE_ROWS,
    PUBLIC_CLS_ROW,
    PUBLIC_ROWS,
    SEQUENCE_GROUP_IDS,
    STATE_VALUE_CLS_ROW,
    TARGET_ROWS,
    VALUE_CLS_ROW,
)
from rl.model.encoder import Encoder
from rl.model.heads import (
    CategoricalValueLogitHead,
    FlatActionReadout,
    HeadParams,
    ReadoutRows,
    RegressionValueLogitHead,
    SlotConditioning,
    chosen_bank_rows,
    compute_policy_metrics,
    sample_categorical,
)
from rl.model.trunk import row_homogeneity
from rl.model.utils import get_num_params, sampling_log_policy


def actor_params_view(variables):
    """Project checkpoint variables onto the actor's required modules.

    Historical league snapshots carry different unused learner branches.
    Strip those before JIT dispatch, retaining the complete encoder (including
    setup-time parameters) and the value output.
    Indexing required modules deliberately fails on incompatible snapshots.
    """
    required = ["encoder", "action_head", "value_head"]
    # The optional doubles readout is an actor consumer too.
    if "slot_conditioning" in variables["params"]:
        required.append("slot_conditioning")
    return {"params": {name: variables["params"][name] for name in required}}


class Porygon2PlayerModel(nn.Module):
    cfg: ConfigDict

    def setup(self):
        """Three modules: the trunk, the action readout, the critic.

        The action readout scores the block cells from the rows it owns;
        `value_head` reads the CLS row and nothing else.
        """
        self.encoder = Encoder(self.cfg.encoder)
        self.action_head = FlatActionReadout(self.cfg.action_head, name="action_head")
        self.value_head = CategoricalValueLogitHead(self.cfg.value_head)
        # The privileged critic (2026-09-01): architecturally identical to
        # value_head, reading VALUE_CLS -- the one row that attends over the
        # opponent-truth partition. Called only under cfg.train, so its
        # params exist in the learner-initialised tree and an actor apply
        # never visits them; nothing at deploy consumes its output.
        self.privileged_value_head = CategoricalValueLogitHead(
            self.cfg.privileged_value_head
        )
        # The public critic (2026-09-15): the same head over PUBLIC_CLS, a
        # value of the common-knowledge state alone. Learner-only likewise.
        self.public_value_head = CategoricalValueLogitHead(self.cfg.public_value_head)
        # The state value head (2026-09-20): the same head over STATE_VALUE_CLS,
        # a value read through the 15 public state rows. Learner-only likewise.
        self.state_value_head = CategoricalValueLogitHead(self.cfg.state_value_head)
        # The consequence model (2026-09-20): learner-only, singles only.
        if self.cfg.train and self.cfg.num_decision_slots == 1:
            self.consequence = ConsequenceModel(self.cfg.consequence)
        # The PBRS potential channel's value (2026-09-11): learner-only, and
        # absent unless the channel runs, so strength 0 keeps today's tree.
        if self.cfg.potential_head.enabled:
            self.potential_head = RegressionValueLogitHead(self.cfg.potential_head)
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
        sequence_rows: ReadoutRows,
        valid_mask: jax.Array,
        head: PolicyHeadOutput,
        train: bool,
        temp: float,
        greedy: bool = False,
    ):
        """Dispatch on decision slots: singles = one flat categorical over
        the block cells (the historical path, unchanged); doubles = two head-level
        stages over per-slot masks with slot 2 conditioned on slot 1's
        choice — the trunk is forwarded once either way."""
        if self.cfg.num_decision_slots == 2:
            return self._forward_two_slots(
                sequence_rows, valid_mask, head, train, temp, greedy
            )
        return self._forward_single_slot(
            sequence_rows, valid_mask, head, train, temp, greedy
        )

    def _legal_logits(
        self,
        sequence_rows: ReadoutRows,
        valid_mask: jax.Array,
        temp: float,
        decision_slot: int = 0,
    ) -> jax.Array:
        """The readout's logits with illegal cells at -1e9 (finite, so no
        `-inf * 0` in a vjp): the one form the sampler reads.
        `decision_slot` picks the ally row the switch
        block reads -- 0 in singles, 1 for doubles stage 2."""
        logits = self.action_head(sequence_rows, temp=temp, decision_slot=decision_slot)
        return jnp.where(valid_mask, logits, -1e9)

    def _score_and_sample(
        self,
        sequence_rows: ReadoutRows,
        valid_mask: jax.Array,
        given_index: jax.Array | None,
        temp: float,
        greedy: bool = False,
        decision_slot: int = 0,
    ):
        """Score one decision's cells and pick an action.

        THE policy scoring path — singles calls it once, doubles calls it
        once per stage with shared params. `given_index` teacher-forces the
        stored choice so the learner's recompute conditions on what the
        actor actually did; None samples.

        Behaviour policy mu == pi, with illegal cells at the dtype's min so
        the sampler can never draw one. `greedy` collapses mu ONLY onto
        pi's most likely legal cell (rl/model/utils.py
        sampling_log_policy; False is bit-identical): the metrics read pi
        untouched, and the stored log_prob is mu's, so an eval slot
        playing the argmax reports what it played.
        """
        flat_valid = valid_mask
        pi_logits = self._legal_logits(sequence_rows, valid_mask, temp, decision_slot)
        # prior=None is uniform over legal cells -- which is exactly what the
        # flat readout's all-zero init produces, so the init policy and the
        # metric anchor are the same distribution.
        metrics = compute_policy_metrics(logits=pi_logits, valid_mask=flat_valid)
        log_mu = sampling_log_policy(metrics.log_policy, flat_valid, greedy)
        if given_index is not None:
            action_index = given_index
        else:
            action_index = sample_categorical(log_mu, self.make_rng("sampling"))
        log_prob = jnp.take(log_mu, action_index, axis=-1)
        return flat_valid, metrics, action_index, log_prob

    def _forward_single_slot(
        self,
        sequence_rows: ReadoutRows,
        valid_mask: jax.Array,
        head: PolicyHeadOutput,
        train: bool,
        temp: float,
        greedy: bool = False,
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
            greedy,
        )
        learner_only = {}
        if self.cfg.train:
            modality_log_probs, valid_per_modality = self._modality_log_marginal(
                metrics.log_policy, flat_valid
            )
            learner_only = {
                "log_policy": metrics.log_policy,
                "normalized_modality_entropy": self._calculate_entropy_metrics(
                    modality_log_probs, valid_per_modality
                ),
            }
        return PlayerPolicyHeadOutput(
            action_index=action_index,
            log_prob=log_prob,
            **learner_only,
            entropy=metrics.entropy,
            normalized_entropy=metrics.normalized_entropy,
            magnet_kl=metrics.magnet_kl,
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
        sequence_rows: ReadoutRows,
        valid_mask: jax.Array,
        head: PolicyHeadOutput,
        train: bool,
        temp: float,
        greedy: bool = False,
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
            sequence_rows, valid_mask[0], stage1_given, temp, None, greedy
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
            greedy,
            decision_slot=1,
        )

        action_index = jnp.stack([index_1, index_2])

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

        learner_only = {}
        if self.cfg.train:
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
            learner_only = {
                "log_policy": jnp.stack([metrics_1.log_policy, metrics_2.log_policy]),
                "normalized_modality_entropy": normalized_modality_entropy,
            }
        return PlayerPolicyHeadOutput(
            action_index=action_index,
            log_prob=log_prob_1 + log_prob_2,
            **learner_only,
            entropy=entropy,
            normalized_entropy=normalized_entropy,
            magnet_kl=metrics_1.magnet_kl + metrics_2.magnet_kl,
        )

    @staticmethod
    def readout_rows(
        sequence: jax.Array, row_valid: jax.Array, env_step: PlayerEnvOutput
    ) -> ReadoutRows:
        """The action readout's rows, sliced by name. Presence, life and
        being on the field are read from each entity's own public row, so
        the pair terms are live on a forced switch and at team preview,
        where no enemy TARGET row is valid."""
        fainted = (
            env_step.public_team[
                :, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__FAINTED
            ]
            != 0
        )
        active = (
            env_step.public_team[
                :, EntityPublicNodeFeature.ENTITY_PUBLIC_NODE_FEATURE__ACTIVE
            ]
            > 0
        )
        public_valid = row_valid[PUBLIC_ROWS]
        return ReadoutRows(
            private=sequence[PRIVATE_ROWS],
            move=sequence[MOVE_ROWS],
            target=sequence[TARGET_ROWS],
            public=sequence[PUBLIC_ROWS],
            public_alive=public_valid & ~fainted,
            public_active=public_valid & active,
        )

    def get_head_outputs(
        self,
        sequence: jax.Array,
        row_valid: jax.Array,
        trunk_group_stats: tuple[jax.Array, jax.Array, jax.Array] | None,
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
            self.readout_rows(sequence, row_valid, env_step),
            env_step.action_mask,
            actor_output.action_head,
            train=self.cfg.train,
            temp=head_params.temp,
            greedy=head_params.greedy,
        )
        learner_only = {}
        if self.cfg.train:
            # Rows converging to one direction reads on the existing panels
            # as "entropy at ceiling while the pointer params grow" -- the
            # phase-1 support-anchor shape -- so it gets its own reading.
            # Offline twin: rl/probes/trunk_homogeneity.py, per block.
            row_cosine, row_participation = row_homogeneity(sequence)
            group_l2_sum, group_rows, in_out_cosine_sum = trunk_group_stats
            learner_only = {
                # The privileged critic: VALUE_CLS, and only VALUE_CLS.
                "priv_value_head": self.privileged_value_head(sequence[VALUE_CLS_ROW]),
                "public_value_head": self.public_value_head(sequence[PUBLIC_CLS_ROW]),
                "state_value_head": self.state_value_head(
                    sequence[self.encoder.local_row(STATE_VALUE_CLS_ROW)]
                ),
                "trunk_row_cosine": row_cosine,
                "trunk_row_participation": row_participation,
                "trunk_out_group_l2_sum": group_l2_sum,
                "trunk_out_group_rows": group_rows,
                "trunk_in_out_cosine_sum": in_out_cosine_sum,
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
            if self.cfg.num_decision_slots == 1:
                learner_only["consequence_inputs"] = self.consequence_inputs(
                    sequence, row_valid, actor_output.action_head.action_index
                )
                pair_features = self.action_head.chosen_pair_features(
                    self.readout_rows(sequence, row_valid, env_step),
                    actor_output.action_head.action_index,
                )
                learner_only["consequence_inputs"] = learner_only[
                    "consequence_inputs"
                ].replace(action_features=pair_features)
                if self.is_initializing():
                    # A submodule's params exist only once it has been called,
                    # and the learner's forward never calls this one.
                    self.consequence(
                        learner_only["consequence_inputs"],
                        jnp.zeros(CONSEQUENCE_NOISE_SIZE, sequence.dtype),
                    )
                    self.consequence.observable(pair_features)
        return PlayerActorOutput(
            action_head=action_head,
            # The CLS row, and only the CLS row.
            value_head=self.value_head(sequence[CLS_ROW]),
            history_carry=history_carry,
            **learner_only,
        )

    def consequence_inputs(
        self, sequence: jax.Array, row_valid: jax.Array, action_cell: jax.Array
    ) -> ConsequenceInputs:
        source_row, target_row = chosen_bank_rows(
            sequence[PRIVATE_ROWS],
            sequence[MOVE_ROWS],
            sequence[TARGET_ROWS],
            action_cell,
        )
        return ConsequenceInputs(
            state_rows=sequence[CONSEQUENCE_ROWS],
            state_valid=row_valid[CONSEQUENCE_ROWS],
            source_row=source_row,
            target_row=target_row,
            cls_row=sequence[CLS_ROW],
        )

    def consequences(
        self, inputs: ConsequenceInputs, noise: jax.Array
    ) -> ConsequenceOutput:
        """One decision, one noise draw. A method of its own so the learner
        applies it separately from `__call__`.

        The heads emit a CHANGE; what is returned is the predicted NEXT rows in
        the trunk's own output form -- current rows plus the change, through
        the trunk's output normalisation (2026-09-20). A prediction is thereby
        constrained to the manifold real rows live on, so it can be scored
        against the real next rows at a FIXED scale (the live per-batch
        normaliser it replaced was gamed by the trunk inflating its rows'
        step-to-step change), and the state value head reads a predicted row
        that looks like the rows it was trained on. The current rows enter
        under stop_gradient: gradient reaches the trunk through the heads'
        inputs, never through an identity path from `now` to the prediction."""
        changes = self.consequence(inputs, noise)
        now = jax.lax.stop_gradient(inputs.state_rows)
        group_ids = jnp.asarray(SEQUENCE_GROUP_IDS[CONSEQUENCE_ROWS])

        def as_trunk_output(change: jax.Array) -> jax.Array:
            return self.encoder.output_normalisation.moved(
                now, change, inputs.state_valid, group_ids
            )

        def implied_value(predicted_rows: jax.Array) -> jax.Array:
            value_row = jax.lax.stop_gradient(predicted_rows[-1])
            return self.state_value_head(value_row).expectation

        mean = as_trunk_output(changes.mean)
        sample = as_trunk_output(changes.sample)
        return ConsequenceOutput(
            mean=mean,
            state_only_mean=as_trunk_output(changes.state_only_mean),
            sample=sample,
            mean_value=implied_value(mean),
            sample_value=implied_value(sample),
            observable_logits=self.consequence.observable(inputs.action_features),
        )

    def __call__(
        self,
        actor_input: PlayerActorInput,
        actor_output: PlayerActorOutput,
        head_params: HeadParams,
    ):
        (
            sequence,
            row_valid,
            trunk_group_stats,
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
            trunk_group_stats,
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
