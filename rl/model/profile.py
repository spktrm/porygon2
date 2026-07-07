import flax.linen as nn
import jax

from inference.model import get_player_model_config
from rl.actor.agent import PlayerActorOutput
from rl.environment.utils import get_ex_player_step
from rl.learner import checkpoint
from rl.model.heads import HeadParams
from rl.model.player_model import get_player_model


def main(generation: int = 9):
    actor_network = get_player_model(get_player_model_config(generation, train=False))
    learner_network = get_player_model(get_player_model_config(generation, train=True))

    ex_actor_input, ex_actor_output = jax.device_put(
        jax.tree.map(lambda x: x[:, 0], get_ex_player_step())
    )
    key = jax.random.key(42)

    latest_ckpt = checkpoint.most_recent_ckpt_dir(f"./ckpts/gen{generation}")
    if latest_ckpt:
        print(f"loading checkpoint from {latest_ckpt}")
        params = checkpoint.load_component(latest_ckpt, "player", "params")
    else:
        params = learner_network.init(
            key, ex_actor_input, ex_actor_output, HeadParams()
        )

    nn.enable_named_call()

    with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        actor_output = actor_network.apply(
            params,
            ex_actor_input,
            PlayerActorOutput(),
            HeadParams(temp=0.8),
            rngs={"sampling": key},
        )

        jax.block_until_ready(actor_output)


if __name__ == "__main__":
    main()
