import functools

import jax

from rl.environment.utils import get_ex_player_step
from rl.learner import checkpoint
from rl.model.config import get_player_model_config
from rl.model.heads import HeadParams
from rl.model.player_model import get_player_model


def main(generation: int = 9):
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

    f = functools.partial(learner_network.apply, params)

    z = jax.jit(f).lower(ex_actor_input, ex_actor_output, HeadParams())

    print(z.compile().cost_analysis())


if __name__ == "__main__":
    main()
