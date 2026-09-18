from dotenv import load_dotenv

load_dotenv()
import functools
from pprint import pprint

import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from rl.environment.data import (
    CAT_VF_SUPPORT,
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
from rl.environment.protos.features_pb2 import EntityPublicNodeFeature
from rl.environment.utils import get_ex_player_step
from rl.model import event_search
from rl.model.config import get_player_model_config
from rl.model.constants import (
    CLS_ROW,
    MOVE_ROWS,
    PRIVATE_ROWS,
    PUBLIC_CLS_LOCAL_ROW,
    PUBLIC_CLS_ROW,
    PUBLIC_ROWS,
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
    compute_policy_metrics,
    sample_categorical,
)
from rl.model.trunk import row_homogeneity
from rl.model.utils import get_num_params, prune_log_policy
from rl.model.world_model import EventWorldModel


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
    # The searching eval actor reads the world model and the public critic.
    for name in ("world_model", "public_value_head"):
        if name in variables["params"]:
            required.append(name)
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
        # The PBRS potential channel's value (2026-09-11): learner-only, and
        # absent unless the channel runs, so strength 0 keeps today's tree.
        if self.cfg.potential_head.enabled:
            self.potential_head = RegressionValueLogitHead(self.cfg.potential_head)
        if self.cfg.world_model.enabled:
            self.world_model = EventWorldModel(self.cfg.world_model, name="world_model")
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
        prune_threshold: float = 0.0,
        logit_bonus: jax.Array | None = None,
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
            sequence_rows, valid_mask, head, train, temp, prune_threshold, logit_bonus
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
        prune_threshold: float = 0.0,
        decision_slot: int = 0,
        logit_bonus: jax.Array | None = None,
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
        if logit_bonus is not None:
            pi_logits = jnp.where(valid_mask, pi_logits + logit_bonus, pi_logits)
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
        sequence_rows: ReadoutRows,
        valid_mask: jax.Array,
        head: PolicyHeadOutput,
        train: bool,
        temp: float,
        prune_threshold: float = 0.0,
        logit_bonus: jax.Array | None = None,
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
            logit_bonus=logit_bonus,
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
        sequence_rows: ReadoutRows,
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

    def _search_bonus(
        self, sequence: jax.Array, row_valid: jax.Array, env_step: PlayerEnvOutput
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Depth-1 sampled event rollouts (rl/model/event_search.py) from
        this request's public rows: one declared action per enumerated
        legal cell, Q from the public critic at the rollouts' leaves, the
        bonus Q / temp on the legal cells (zero on the value-blind arm)."""
        cfg = self.cfg.search
        public_index = jnp.asarray(self.encoder.search_public_rows())
        public_rows = sequence[public_index]
        public_valid = row_valid[public_index]
        legal = env_step.action_mask
        cells = jnp.nonzero(legal, size=cfg.max_cells, fill_value=0)[0]
        num_legal = legal.sum()
        cell_valid = jnp.arange(cfg.max_cells) < num_legal
        overflow = num_legal > cfg.max_cells
        root = event_search.root_info(env_step)
        declared_kind, declared_arg = jax.vmap(
            functools.partial(
                event_search.declared_from_cell, env_step=env_step, root=root
            )
        )(cells)
        world = self.world_model
        if self.is_initializing():
            # Parameters are created on first use; created inside the
            # rollout's scan they leak as tracers, so touch every reader
            # once outside it at init.
            blank = event_search.blank_tokens()
            world.decode(
                public_rows,
                public_valid,
                jnp.asarray(0),
                jnp.asarray(0),
                blank,
                jnp.asarray(True),
            )
            world.imagine(
                public_rows,
                blank,
                jnp.ones(public_rows.shape[0], jnp.bool_),
                world.delta_scale,
                self.make_rng("sampling"),
            )
            world.terminal_outcome(public_rows)
            self.public_value_head(public_rows[PUBLIC_CLS_LOCAL_ROW])

        def search_fns(module) -> event_search.EventSearchFns:
            """The readers bound to `module` -- the transformed module
            inside a lifted scan, `self` outside it."""
            model = module.world_model

            def value_fn(rows):
                return module.public_value_head(rows[PUBLIC_CLS_LOCAL_ROW]).expectation

            def terminal_fn(rows):
                probs = jax.nn.softmax(model.terminal_outcome(rows))
                return probs @ jnp.asarray(CAT_VF_SUPPORT, jnp.float32)

            return event_search.EventSearchFns(
                decode_fn=lambda rows, kind, arg, tokens, mine: model.decode(
                    rows, public_valid, kind, arg, tokens, mine
                ),
                imagine_fn=lambda rows, tokens, mask, rng: model.imagine(
                    rows, tokens, mask, model.delta_scale, rng
                ),
                value_fn=value_fn,
                terminal_fn=terminal_fn,
            )

        budget = event_search.RolloutBudget(max_events=cfg.max_events, temp=1.0)

        def lifted_scan(step, init, keys):
            def body(module, carry, key):
                return step(search_fns(module), carry, key)

            return nn.scan(
                body,
                variable_broadcast="params",
                split_rngs={"params": False},
                in_axes=0,
                out_axes=0,
            )(self, init, keys)

        q, _ = event_search.q_values(
            public_rows,
            declared_kind,
            declared_arg,
            cell_valid,
            root,
            search_fns(self),
            budget,
            cfg.num_samples,
            self.make_rng("sampling"),
            scan=lifted_scan,
        )
        cell_bonus = event_search.search_bonus(q, cell_valid, cfg.temp, cfg.value_blind)
        bonus = (
            jnp.zeros(legal.shape, jnp.float32)
            .at[cells]
            .add(jnp.where(cell_valid, cell_bonus, 0.0))
        )
        bonus = jnp.where(overflow, 0.0, bonus)
        base = self._legal_logits(
            self.readout_rows(sequence, row_valid, env_step), legal, 1.0
        )
        diagnostics = event_search.search_diagnostics(base, bonus, legal)
        return bonus, {
            "search_root_kl": diagnostics["search_root_kl"],
            "search_bonus_gap": diagnostics["search_bonus_gap"],
            "search_overflow": overflow,
        }

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
        logit_bonus = None
        search_outputs = {}
        if self.cfg.search.enabled:
            logit_bonus, search_outputs = self._search_bonus(
                sequence, row_valid, env_step
            )
        action_head = self._forward_action_head(
            self.readout_rows(sequence, row_valid, env_step),
            env_step.action_mask,
            actor_output.action_head,
            train=self.cfg.train,
            temp=head_params.temp,
            prune_threshold=head_params.prune_threshold,
            logit_bonus=logit_bonus,
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
        return PlayerActorOutput(
            action_head=action_head,
            # The CLS row, and only the CLS row.
            value_head=self.value_head(sequence[CLS_ROW]),
            history_carry=history_carry,
            **search_outputs,
            **learner_only,
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
