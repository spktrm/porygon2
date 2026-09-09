"""Decision accounting for overlapping chunks, independent of padded frames."""

import numpy as np


def count_chunk_decisions(done) -> int:
    """Count acted rows before terminal, excluding the final bootstrap row.

    Forced single-option actions count too. Both perspectives count separately;
    this measures player decisions, not unique simulator turns or games.
    """
    done = np.asarray(done, dtype=bool)
    return int((np.cumsum(done, axis=0)[:-1] == 0).sum())
