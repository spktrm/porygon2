from dotenv import load_dotenv

load_dotenv()
import functools
from pprint import pprint

import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from rl.environment.data import FLAT_MODALITY_MASK, NUM_MODALITY_FEATURES
from rl.environment.interfaces import (
    PlayerActorInput,
    PlayerActorOutput,
    PlayerEnvOutput,
    PlayerPolicyHeadOutput,
    PolicyHeadOutput,
)
from rl.environment.protos.service_pb2 import ModalityEnum
from rl.environment.utils import get_ex_player_step
from rl.model.config import get_player_model_config
from rl.model.encoder import Encoder
from rl.model.heads import (
    ActionAdapter,
    CategoricalValueLogitHead,
    HeadParams,
    MacroMicroHead,
    SlotConditioning,
    calculate_hierarchical_prior,
    compose_action_grid,
    compute_policy_metrics,
    sample_categorical,
)
from rl.model.utils import get_num_params, legal_log_policy


def _sampling_log_policy(log_policy: jax.Array, valid_mask: jax.Array) -> jax.Array:
    """log pi with illegal cells at the dtype's min, for sample_categorical."""
    return jnp.where(valid_mask, log_policy, jnp.finfo(log_policy.dtype).min)


class Porygon2PlayerModel(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.encoder = Encoder(self.cfg.encoder)
        # Typed action streams (2026-08-17): the trunk carries move /
        # switch / target slots as separate residual streams, so the
        # within-modality readout is a parameter-less dot grid and the
        # modality head pools typed spaces — depth lives in the trunk,
        # not in per-modality head stacks. Since 2026-08-20 the macro +
        # micro pair is one shared MacroMicroHead (the Q critic
        # instantiates the same module below), read through an owned
        # ActionAdapter: zero-init residual, exact identity at init/fresh
        # reload, so the parameter-free micro dot grid no longer rides
        # directly on embeddings the Q head's CE is also shaping.
        self.macro_micro_head = MacroMicroHead(self.cfg.macro_micro)
        self.policy_adapter = ActionAdapter(self.cfg.policy_adapter)
        self.v_head = CategoricalValueLogitHead(self.cfg.v_head)
        # Counterfactual value ladder (2026-08-16): all/private/public are
        # independent estimators per information route (separate query
        # inits and residual gates in the trunk; shared read module). The
        # private rung (deployable information set — no opponent sheet)
        # still reads through the SAME v_head as the privileged main
        # readout — a deliberate remaining coupling that keeps the two
        # rungs' output calibration comparable; public (history-context-
        # only) gets its own head.
        self.public_v_head = CategoricalValueLogitHead(self.cfg.v_head)
        if self.cfg.num_decision_slots == 2:
            # Doubles only: params appear in the tree only when the module
            # is called, so singles checkpoints are unaffected; a future
            # doubles resume via load-mode "params" fresh-inits this.
            self.slot_conditioning = SlotConditioning()
        # Privileged two-rung Q critic (docs/q-critic-plan.md), STRUCTURAL
        # since 2026-08-20 — no enable flag, every consumer (Retrace,
        # boost, COMA, diagnostics) assumes it exists. Same module stack
        # as the policy on the critic side: an owned ActionAdapter (the
        # projected value conditioning concatenated in, so the rung's
        # information set reaches every CELL, not just the macro level)
        # into the shared MacroMicroHead at num_logits = bin count,
        # composed via compose_action_grid. The one module pair serves
        # both rungs — called with the privileged value_all embedding
        # (Q_all, drives Retrace) and the private value embedding
        # (Q_private, the policy's information set) — sharing every
        # param, the same calibration coupling as v_head's private
        # readout. Learner-only (train gate in __call__); never sampled
        # from.
        self.q_cond_proj = nn.Dense(self.cfg.entity_size)
        self.q_cond_norm = nn.LayerNorm()
        self.q_adapter = ActionAdapter(self.cfg.q_head.adapter)
        self.q_macro_micro = MacroMicroHead(self.cfg.q_head.macro_micro)

    def _calculate_entropy_metrics(
        self, policy_metrics: PolicyHeadOutput, flat_valid_mask: jax.Array
    ):
        modality_oh = jax.nn.one_hot(
            FLAT_MODALITY_MASK,
            NUM_MODALITY_FEATURES,
            dtype=policy_metrics.log_policy.dtype,
        )
        valid_modality_mask = flat_valid_mask[..., None] * modality_oh

        modality_log_probs = nn.logsumexp(
            jnp.where(
                valid_modality_mask,
                policy_metrics.log_policy[..., None],
                -1e9,
            ),
            axis=0,
        )
        modality_probs = jnp.exp(modality_log_probs)

        # Count valid actions per modality
        valid_actions_per_modality = valid_modality_mask.sum(axis=0)

        # --- THE FIX ---

        # 1. Count how many total modalities actually have valid options
        num_valid_modalities = (valid_actions_per_modality > 0).sum(
            dtype=modality_probs.dtype
        )

        # 2. Calculate the raw entropy safely
        raw_modality_entropy = -jnp.sum(
            jnp.where(
                valid_actions_per_modality > 0, modality_probs * modality_log_probs, 0.0
            )
        )

        # 3. Calculate max possible entropy
        max_modality_entropy = jnp.log(jnp.maximum(num_valid_modalities, 1.0))

        # --- THE FIX ---
        # Create a safe denominator that is never 0.0, even when the mask is False
        safe_max_modality_entropy = jnp.where(
            num_valid_modalities > 1, max_modality_entropy, 1.0
        )

        # 4. Safely normalize using the safe denominator
        return jnp.where(
            num_valid_modalities > 1,
            raw_modality_entropy / safe_max_modality_entropy,
            0.0,
        )

    def _forward_action_head(
        self,
        action_embeddings: jax.Array,
        valid_mask: jax.Array,
        head: PolicyHeadOutput,
        train: bool,
        temp: float,
    ):
        """Dispatch on decision slots: singles = one flat categorical over
        the grid (the historical path, unchanged); doubles = two head-level
        stages over per-slot masks with slot 2 conditioned on slot 1's
        choice — the trunk is forwarded once either way."""
        if self.cfg.num_decision_slots == 2:
            return self._forward_two_slots(
                action_embeddings, valid_mask, head, train, temp
            )
        return self._forward_single_slot(
            action_embeddings, valid_mask, head, train, temp
        )

    def _forward_single_slot(
        self,
        action_embeddings: jax.Array,
        valid_mask: jax.Array,
        head: PolicyHeadOutput,
        train: bool,
        temp: float,
    ):
        flat_valid_mask = valid_mask.reshape(-1)

        action_embeddings = self.policy_adapter(action_embeddings)
        # A src slot is actionable iff its row has any valid tgt cell.
        src_valid = valid_mask.any(axis=-1)
        macro_logits, square_logits = self.macro_micro_head(
            action_embeddings, src_valid
        )
        square_logits = square_logits / temp

        # Hierarchical composition (compose_action_grid, logsumexp reduce):
        # a macro softmax over modalities times a micro softmax within each
        # modality, multiplied in log space. The macro logits come from a
        # dedicated head over per-modality pooled src embeddings rather
        # than a mean-pool of the square logits, so the gram logits only
        # ever receive within-modality (per-modality shift-invariant)
        # gradient — micro confidence cannot move the modality contest
        # through logit magnitude. The policy gradient still splits into a
        # within-modality term and a modality-level term like the
        # hierarchical multi-head did.
        modality_oh = FLAT_MODALITY_MASK[:, None] == jnp.arange(NUM_MODALITY_FEATURES)
        modality_counts = (flat_valid_mask[:, None] & modality_oh).sum(axis=0)

        macro_logits = macro_logits / temp
        log_macro_policy = legal_log_policy(macro_logits, modality_counts > 0)

        pi_logits = jnp.where(
            flat_valid_mask,
            compose_action_grid(
                log_macro_policy, square_logits, flat_valid_mask, reduce="logsumexp"
            ),
            -1e9,
        )

        prior = calculate_hierarchical_prior(flat_valid_mask)
        policy_metrics = compute_policy_metrics(
            logits=pi_logits, valid_mask=flat_valid_mask, prior=prior
        )
        # Behaviour policy mu == pi; illegal cells carry the dtype's min so
        # the sampler never draws them.
        log_mu = _sampling_log_policy(policy_metrics.log_policy, flat_valid_mask)

        if train:
            action_index = head.action_index
        else:
            action_index = sample_categorical(log_mu, self.make_rng("sampling"))

        log_prob = jnp.take(log_mu, action_index, axis=-1)

        mask_width = valid_mask.shape[-1]
        src_index = action_index // mask_width
        tgt_index = action_index % mask_width

        normalized_modality_entropy = self._calculate_entropy_metrics(
            policy_metrics, flat_valid_mask
        )

        return PlayerPolicyHeadOutput(
            action_index=action_index,
            log_prob=log_prob,
            # Full support only in the learner: the magnet KL needs both
            # distributions; actors skip it so replay transitions stay small.
            log_policy=policy_metrics.log_policy if self.cfg.train else (),
            logits=pi_logits if self.cfg.train else (),
            src_index=src_index,
            tgt_index=tgt_index,
            entropy=policy_metrics.entropy,
            normalized_entropy=policy_metrics.normalized_entropy,
            magnet_kl=policy_metrics.magnet_kl,
            normalized_modality_entropy=normalized_modality_entropy,
        )

    def _score_stage(
        self,
        action_embeddings: jax.Array,
        valid_mask: jax.Array,
        given_index: jax.Array | None,
        temp: float,
    ):
        """One decision stage of the doubles path: score the grid with the
        per-modality heads (params shared across stages), compose with the
        macro head, and pick an action — teacher-forced when given_index is
        provided so the learner's recompute conditions on the stored
        choice."""
        flat_valid_mask = valid_mask.reshape(-1)
        action_embeddings = self.policy_adapter(action_embeddings)
        src_valid = valid_mask.any(axis=-1)
        macro_logits, square_logits = self.macro_micro_head(
            action_embeddings, src_valid
        )
        square_logits = square_logits / temp

        modality_oh = FLAT_MODALITY_MASK[:, None] == jnp.arange(NUM_MODALITY_FEATURES)
        modality_counts = (flat_valid_mask[:, None] & modality_oh).sum(axis=0)

        macro_logits = macro_logits / temp
        log_macro_policy = legal_log_policy(macro_logits, modality_counts > 0)

        pi_logits = jnp.where(
            flat_valid_mask,
            compose_action_grid(
                log_macro_policy, square_logits, flat_valid_mask, reduce="logsumexp"
            ),
            -1e9,
        )
        prior = calculate_hierarchical_prior(flat_valid_mask)
        policy_metrics = compute_policy_metrics(
            logits=pi_logits, valid_mask=flat_valid_mask, prior=prior
        )
        log_mu = _sampling_log_policy(policy_metrics.log_policy, flat_valid_mask)
        if given_index is not None:
            action_index = given_index
        else:
            action_index = sample_categorical(log_mu, self.make_rng("sampling"))
        log_prob = jnp.take(log_mu, action_index, axis=-1)
        return flat_valid_mask, policy_metrics, action_index, log_prob

    def _apply_choice_collision(self, valid_mask: jax.Array, action_index: jax.Array):
        """Slot-2 legality given slot 1's choice: both mons cannot switch
        to the same reserve, so if slot 1 chose a switch, knock out slot
        2's switch cells sharing its target column. Must be applied
        identically at act and learn time or the stored behaviour log-prob
        and the learner's recompute diverge."""
        mask_width = valid_mask.shape[-1]
        flat = valid_mask.reshape(-1)
        cell_modality = jnp.asarray(FLAT_MODALITY_MASK)
        a1_is_switch = (
            jnp.take(cell_modality, action_index) == ModalityEnum.MODALITY_ENUM__SWITCH
        )
        same_tgt = (jnp.arange(flat.shape[0]) % mask_width) == (
            action_index % mask_width
        )
        is_switch_cell = cell_modality == ModalityEnum.MODALITY_ENUM__SWITCH
        collide = a1_is_switch & same_tgt & is_switch_cell
        return jnp.where(collide, False, flat).reshape(valid_mask.shape)

    def _forward_two_slots(
        self,
        action_embeddings: jax.Array,
        valid_mask: jax.Array,
        head: PolicyHeadOutput,
        train: bool,
        temp: float,
    ):
        """Doubles: valid_mask is (2, N, N) per-slot masks and, in train,
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
        action indices, (2, N*N) full-support log_policy in the learner)
        is the remaining doubles workstream; the model side is complete.
        """
        stage1_given = head.action_index[0] if train else None
        flat_valid_1, metrics_1, index_1, log_prob_1 = self._score_stage(
            action_embeddings, valid_mask[0], stage1_given, temp
        )

        mask_width = valid_mask.shape[-1]
        cond_embeddings = self.slot_conditioning(
            action_embeddings, index_1 // mask_width, index_1 % mask_width
        )
        mask_2 = self._apply_choice_collision(valid_mask[1], index_1)
        stage2_given = head.action_index[1] if train else None
        flat_valid_2, metrics_2, index_2, log_prob_2 = self._score_stage(
            cond_embeddings, mask_2, stage2_given, temp
        )

        action_index = jnp.stack([index_1, index_2])
        # Diagnostic average of the per-stage values (a true joint version
        # would need raw/max modality entropies threaded out; not worth it
        # for telemetry).
        normalized_modality_entropy = (
            self._calculate_entropy_metrics(metrics_1, flat_valid_1)
            + self._calculate_entropy_metrics(metrics_2, flat_valid_2)
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
            src_index=action_index // mask_width,
            tgt_index=action_index % mask_width,
            entropy=entropy,
            normalized_entropy=normalized_entropy,
            magnet_kl=metrics_1.magnet_kl + metrics_2.magnet_kl,
            normalized_modality_entropy=normalized_modality_entropy,
        )

    def _forward_value_head(self, value_embeddings: jax.Array):
        """value_embeddings: (4 * entity_size,)."""
        return self.v_head(value_embeddings)

    def _forward_q_head(
        self, action_embeddings: jax.Array, cond: jax.Array, valid_mask: jax.Array
    ) -> jax.Array:
        """One Q rung over the flat action grid: (..., N * N, n_bins)
        categorical logits. cond is the rung's pooled value embedding —
        the information set (value_all = privileged Q_all, private value
        = deployable Q_private). Projection/norm in f32 (flax default),
        cast back so the bf16 grid tensors stay bf16.
        """
        c = self.q_cond_norm(self.q_cond_proj(cond)).astype(action_embeddings.dtype)
        adapted = self.q_adapter(action_embeddings, cond=c)
        src_valid = valid_mask.any(axis=-1)
        macro, micro = self.q_macro_micro(adapted, src_valid, cond=c)
        flat_valid_mask = valid_mask.reshape(*valid_mask.shape[:-2], -1)
        return compose_action_grid(macro, micro, flat_valid_mask, reduce="mean")

    def get_head_outputs(
        self,
        action_embeddings: jax.Array,
        value_embeddings: jax.Array,
        env_step: PlayerEnvOutput,
        actor_output: PlayerActorOutput,
        head_params: HeadParams,
    ):

        action_head = self._forward_action_head(
            action_embeddings,
            env_step.action_mask,
            actor_output.action_head,
            train=self.cfg.train,
            temp=head_params.temp,
        )

        return PlayerActorOutput(
            action_head=action_head,
            value_head=self._forward_value_head(value_embeddings),
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
            action_embeddings,
            value_embeddings,
            private_value_embeddings,
            public_value_embeddings,
        ) = self.encoder(
            actor_input.env, actor_input.packed_history, actor_input.history
        )

        outputs = jax.vmap(
            functools.partial(self.get_head_outputs, head_params=head_params)
        )(action_embeddings, value_embeddings, actor_input.env, actor_output)

        if self.cfg.train:
            # Counterfactual value ladder, learner-only so replay
            # transitions stay small. private reads the SHARED v_head —
            # see setup.
            outputs = outputs.replace(
                private_value_logits=self.v_head(private_value_embeddings).logits,
                public_value_logits=self.public_v_head(public_value_embeddings).logits,
            )
            # Two-rung all-action Q readout over the flat action grid —
            # (T, N*N, n_bins) categorical logits per rung. Q_all is
            # privileged via the value_all conditioning and drives the
            # Retrace recursion; Q_private shares every param but sees
            # only the policy's information set. Retrace targets and
            # diagnostics live learner-side; nothing here feeds the
            # policy (docs/q-critic-plan.md).
            outputs = outputs.replace(
                q_logits=self._forward_q_head(
                    action_embeddings,
                    value_embeddings,
                    actor_input.env.action_mask,
                ),
                private_q_logits=self._forward_q_head(
                    action_embeddings,
                    private_value_embeddings,
                    actor_input.env.action_mask,
                ),
            )
        return outputs


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
