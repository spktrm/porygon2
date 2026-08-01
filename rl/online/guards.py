"""Pure decision helpers for actor data routing.

Kept deliberately dependency-free (no jax / env imports) so the rules that keep
evaluation trajectories out of the training data can be unit-tested in isolation.
"""

TRAIN_USERNAME_PREFIX = "train"


def should_push_trajectory(is_eval: bool, do_push: bool, username: str) -> bool:
    """Whether an unrolled trajectory may enter the training replay buffer.

    Evaluation trajectories must never become training data. The decision is
    gated primarily on the explicit ``is_eval`` flag set when the actor is
    constructed, so renaming an actor (or adding a new one) cannot silently leak
    eval games into training. The ``train`` username prefix is retained as a
    defensive cross-check rather than as the sole safeguard.
    """
    if is_eval:
        return False
    if not do_push:
        return False
    return username.startswith(TRAIN_USERNAME_PREFIX)
