"""Shared test setup.

Env vars must be set before jax/wandb are imported anywhere: tests run on
the training box, so JAX must not preallocate the GPU out from under a
live learner (see no-agent-testing memory) and wandb must never try to
sync.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TQDM_DISABLE", "1")
