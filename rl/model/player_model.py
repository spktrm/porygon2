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
from rl.environment.utils import get_ex_player_step
from rl.model.config import get_player_model_config
from rl.model.encoder import Encoder
from rl.model.heads import (
    CategoricalValueLogitHead,
    HeadParams,
    MacroHead,
    PairPolicyHead,
    calculate_hierarchical_prior,
    compute_policy_metrics,
    sample_categorical,
)
from rl.model.utils import get_num_params, legal_log_policy


class Porygon2PlayerModel(nn.Module):
    cfg: ConfigDict

    def setup(self):
        self.encoder = Encoder(self.cfg.encoder)
        self.pi_head = PairPolicyHead(self.cfg.pi_head)
        self.macro_head = MacroHead(self.cfg.macro_head)
        self.v_head = CategoricalValueLogitHead(self.cfg.v_head)

    def _forward_pi_head(self, action_embeddings: jax.Array):
        """Untied src x tgt pointer logits.

        action_embeddings: (NUM_ACTION_FEATURES, entity_size), already
        normed by the encoder's out-norms. Returns
        (NUM_ACTION_FEATURES**2,) src x tgt logits.
        """
        return self.pi_head(action_embeddings)

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
        flat_valid_mask = valid_mask.reshape(-1)

        square_logits = self._forward_pi_head(action_embeddings) / temp

        # Hierarchical composition: a macro softmax over modalities times a
        # micro softmax within each modality, multiplied in log space. The
        # macro logits come from a dedicated head over per-modality pooled
        # src embeddings rather than a mean-pool of the square logits, so
        # the gram logits only ever receive within-modality (per-modality
        # shift-invariant) gradient — micro confidence cannot move the
        # modality contest through logit magnitude. The policy gradient
        # still splits into a within-modality term and a modality-level
        # term like the hierarchical multi-head did.
        modality_oh = FLAT_MODALITY_MASK[:, None] == jnp.arange(NUM_MODALITY_FEATURES)
        valid_per_modality = flat_valid_mask[:, None] & modality_oh
        modality_counts = valid_per_modality.sum(axis=0)

        micro_lse = nn.logsumexp(
            jnp.where(valid_per_modality, square_logits[:, None], -1e9), axis=0
        )
        log_micro_policy = square_logits - micro_lse[FLAT_MODALITY_MASK]

        # A src slot is actionable iff its row has any valid tgt cell.
        src_valid = valid_mask.any(axis=-1)
        macro_logits = self.macro_head(action_embeddings, src_valid) / temp
        log_macro_policy = legal_log_policy(macro_logits, modality_counts > 0)

        pi_logits = jnp.where(
            flat_valid_mask,
            log_macro_policy[FLAT_MODALITY_MASK] + log_micro_policy,
            -1e9,
        )

        policy_metrics = compute_policy_metrics(
            logits=pi_logits,
            valid_mask=flat_valid_mask,
            prior=calculate_hierarchical_prior(flat_valid_mask),
        )

        if train:
            action_index = head.action_index
        else:
            action_index = sample_categorical(pi_logits, self.make_rng("sampling"))

        log_prob = jnp.take(policy_metrics.log_policy, action_index, axis=-1)

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
            src_index=src_index,
            tgt_index=tgt_index,
            entropy=policy_metrics.entropy,
            normalized_entropy=policy_metrics.normalized_entropy,
            magnet_kl=policy_metrics.magnet_kl,
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

        return jax.vmap(
            functools.partial(self.get_head_outputs, head_params=head_params)
        )(action_embeddings, value_embeddings, actor_input.env, actor_output)


def get_player_model(config: ConfigDict = None) -> nn.Module:
    if config is None:
        config = get_player_model_config()
    return Porygon2PlayerModel(config)


def main(generation: int = 9):
    """Init the learner network on an example step and print param counts.

    Attention-map dumps and cost analysis live in rl.model.viz.
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