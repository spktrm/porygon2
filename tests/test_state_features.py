"""`rl/model/state_features.py` (2026-09-04): the three state linears'
input layouts written ONCE, so telemetry's column blocks are derived from
the same list the encoder concatenates. Pins that the blocks partition
each kernel's width exactly and that the hp rows are where the layout
says they are -- a positive control for the kernel-rms panels, which
would otherwise read a wrong block silently.
"""

import numpy as np

from rl.model.state_features import (
    STATE_KERNEL_GROUPS,
    STATE_KERNELS,
    hp_input_rows,
    state_kernel_blocks,
)


def test_blocks_partition_every_kernel_width():
    blocks = state_kernel_blocks()
    assert set(blocks) == set(STATE_KERNELS)
    for kernel, groups in blocks.items():
        assert set(groups) == set(STATE_KERNEL_GROUPS) | {"other"}
        covered = np.concatenate(
            [
                np.arange(block.start, block.stop)
                for group in groups.values()
                for block in group
            ]
        )
        width = max(block.stop for group in groups.values() for block in group)
        assert sorted(covered.tolist()) == list(range(width)), kernel
    # The transient kernel carries no hp block; its hp rows are empty.
    assert len(hp_input_rows("public_transient_linear")) == 0


def test_hp_rows_are_the_hp_scalar_and_its_bins():
    # The hp block is the scalar plus 32 bins (33 columns); the public
    # persistent row also carries the 33-way divided one-hot of HP_RATIO
    # inside its persistent code, and both count as hp.
    private_rows = hp_input_rows("private_state_linear")
    assert len(private_rows) == 33
    assert len(np.unique(private_rows)) == 33
    public_rows = hp_input_rows("public_persistent_linear")
    assert len(public_rows) == 66
    assert len(np.unique(public_rows)) == 66
