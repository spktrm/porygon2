"""The interrupt checkpoint used to be skipped because Ctrl-C landed inside
the jitted step, after the old train state was donated. DeferredInterrupt
holds the first SIGINT until the loop's safe point; the second one aborts.
Runs on pytest's main thread, where the handler can be installed."""

import signal

import pytest

from rl.online.training.learner import DeferredInterrupt


def test_first_interrupt_waits_for_the_safe_point() -> None:
    before = signal.getsignal(signal.SIGINT)
    with DeferredInterrupt() as interrupt:
        assert signal.getsignal(signal.SIGINT) == interrupt._handle
        signal.raise_signal(signal.SIGINT)
        # Still here: the step in flight would have finished.
        assert interrupt.pending
        with pytest.raises(KeyboardInterrupt):
            interrupt.check()
    assert signal.getsignal(signal.SIGINT) == before


def test_second_interrupt_aborts_immediately() -> None:
    with DeferredInterrupt() as interrupt:
        signal.raise_signal(signal.SIGINT)
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGINT)
        assert interrupt.pending


def test_check_is_silent_without_an_interrupt() -> None:
    with DeferredInterrupt() as interrupt:
        interrupt.check()
        assert not interrupt.pending
