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
    ActionScoreHead,
    CategoricalValueLogitHead,
    HeadParams,
    SlotConditioning,
    calculate_hierarchical_prior,
    compose_q,
    compute_policy_metrics,
    sample_categorical,
)
from rl.model.utils import get_num_params


def _sampling_log_policy(log_policy: jax.Array, valid_mask: jax.Array) -> jax.Array:
    """log pi with illegal cells at the dtype's min, for sample_categorical."""
    return jnp.where(valid_mask, log_policy, jnp.finfo(log_policy.dtype).min)


class Porygon2PlayerModel(nn.Module):
    cfg: ConfigDict

    def setup(self):
        """Four modules, and two of them are the same class.

        The action axis is read exactly twice — once as a policy, once as an
        advantage — and both readouts are ActionScoreHead over the SAME
        src x tgt grid, differing only in how macro and micro compose
        (`reduce`). The state axis is read once, by v_head. That is the whole
        model surface; everything else is composition.

        The two ActionScoreHeads do NOT share parameters. The advantage
        head's loss would otherwise reshape the live policy geometry
        directly — each family reads the trunk through its own adapter, so
        the trunk still receives both gradients while the head-specific
        geometry stays decoupled.
        """
        self.encoder = Encoder(self.cfg.encoder)
        self.policy_head = ActionScoreHead(
            self.cfg.policy_head, reduce="logsumexp", name="policy_head"
        )
        self.advantage_head = ActionScoreHead(
            self.cfg.advantage_head, reduce="mean", name="advantage_head"
        )
        self.v_head = CategoricalValueLogitHead(self.cfg.v_head)
        if self.cfg.num_decision_slots == 2:
            # Doubles only: params appear in the tree only when the module
            # is called, so singles checkpoints are unaffected.
            self.slot_conditioning = SlotConditioning()

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

        valid_actions_per_modality = valid_modality_mask.sum(axis=0)
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

    def _score_and_sample(
        self,
        action_embeddings: jax.Array,
        valid_mask: jax.Array,
        given_index: jax.Array | None,
        temp: float,
    ):
        """Score one decision grid and pick an action.

        THE policy scoring path — singles calls it once, doubles calls it
        once per stage with shared params. `given_index` teacher-forces the
        stored choice so the learner's recompute conditions on what the
        actor actually did; None samples.

        Behaviour policy mu == pi, with illegal cells at the dtype's min so
        the sampler can never draw one.
        """
        scores = self.policy_head(action_embeddings, valid_mask, temp=temp)
        pi_logits = jnp.where(scores.flat_valid, scores.logits, -1e9)
        metrics = compute_policy_metrics(
            logits=pi_logits,
            valid_mask=scores.flat_valid,
            prior=calculate_hierarchical_prior(scores.flat_valid),
        )
        log_mu = _sampling_log_policy(metrics.log_policy, scores.flat_valid)
        action_index = (
            given_index
            if given_index is not None
            else sample_categorical(log_mu, self.make_rng("sampling"))
        )
        log_prob = jnp.take(log_mu, action_index, axis=-1)
        return scores, metrics, action_index, log_prob

    def _forward_single_slot(
        self,
        action_embeddings: jax.Array,
        valid_mask: jax.Array,
        head: PolicyHeadOutput,
        train: bool,
        temp: float,
    ):
        scores, metrics, action_index, log_prob = self._score_and_sample(
            action_embeddings, valid_mask, head.action_index if train else None, temp
        )
        mask_width = valid_mask.shape[-1]
        learner_only = {}
        if self.cfg.train:
            learner_only = {
                "log_policy": metrics.log_policy,
            }
        return PlayerPolicyHeadOutput(
            action_index=action_index,
            log_prob=log_prob,
            **learner_only,
            src_index=action_index // mask_width,
            tgt_index=action_index % mask_width,
            entropy=metrics.entropy,
            normalized_entropy=metrics.normalized_entropy,
            magnet_kl=metrics.magnet_kl,
            normalized_modality_entropy=self._calculate_entropy_metrics(
                metrics, scores.flat_valid
            ),
        )

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
        scores_1, metrics_1, index_1, log_prob_1 = self._score_and_sample(
            action_embeddings, valid_mask[0], stage1_given, temp
        )
        flat_valid_1 = scores_1.flat_valid

        mask_width = valid_mask.shape[-1]
        cond_embeddings = self.slot_conditioning(
            action_embeddings, index_1 // mask_width, index_1 % mask_width
        )
        mask_2 = self._apply_choice_collision(valid_mask[1], index_1)
        stage2_given = head.action_index[1] if train else None
        scores_2, metrics_2, index_2, log_prob_2 = self._score_and_sample(
            cond_embeddings, mask_2, stage2_given, temp
        )
        flat_valid_2 = scores_2.flat_valid

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
        action_embeddings, value_embeddings = self.encoder(
            actor_input.env, actor_input.packed_history, actor_input.history
        )

        outputs = jax.vmap(
            functools.partial(self.get_head_outputs, head_params=head_params)
        )(action_embeddings, value_embeddings, actor_input.env, actor_output)

        if self.cfg.train:
            # Q = V + A, composed HERE — one place, next to the heads that
            # produce the two terms, rather than reassembled at four call
            # sites in the learner. Learner-only: actors never read Q, so
            # replay transitions stay small.
            #
            # The advantage head runs with T leading (no vmap) — every
            # module on this path takes arbitrary leading batch dims.
            raw = self.advantage_head(
                action_embeddings,
                actor_input.env.action_mask,
                cond=value_embeddings,
            )
            advantage, q = compose_q(
                outputs.value_head.expectation,
                raw.logits,
                outputs.action_head.log_policy,
                raw.flat_valid,
            )
            outputs = outputs.replace(advantage=advantage, q=q)
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
