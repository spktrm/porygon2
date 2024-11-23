from pprint import pprint

import jax
import numpy as np
from tqdm import trange

from ml.arch.config import get_model_cfg
from ml.arch.model import get_model
from ml.learners import vtrace as learner
from ml.utils import get_most_recent_file
from rlenv.main import BatchCollectorV2, BatchSinglePlayerEnvironment


def main():
    learner_config = learner.get_config()
    model_config = get_model_cfg()
    pprint(learner_config)

    network = get_model(model_config)
    evaluation_collector = BatchCollectorV2(network, 4, BatchSinglePlayerEnvironment)

    state = learner.create_train_state(network, jax.random.PRNGKey(42), learner_config)

    latest_ckpt = get_most_recent_file("./ckpts")
    if latest_ckpt:
        state = learner.load(state, latest_ckpt)

    # initialise average returns
    avg_reward = np.zeros(evaluation_collector.game.num_envs)
    num_inits = 200
    bar = trange(0, num_inits)

    for i in bar:
        batch = evaluation_collector.collect_batch_trajectory(state.params)
        avg_reward += (batch.actor.win_rewards[..., 0] * batch.env.valid).sum(0)
        bar.set_description(np.array2string(avg_reward / i))


if __name__ == "__main__":
    main()
