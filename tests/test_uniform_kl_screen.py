import pytest

from rl.probes import uniform_kl_screen


def test_protocol_switch_report_excludes_pivots_forced_and_incomplete(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        uniform_kl_screen.battle_stats,
        "load_moves",
        lambda: {
            "Tackle": {"selfSwitch": False, "cat": "Physical"},
            "U-turn": {"selfSwitch": True, "cat": "Physical"},
        },
    )
    protocol = "\n".join(
        (
            "|player|p1|Alice",
            "|player|p2|Bob",
            "|switch|p1a: First|Pikachu|100/100",
            "|switch|p2a: Rival|Eevee|100/100",
            "|turn|1",
            "|move|p1a: First|U-turn|p2a: Rival",
            "|switch|p1a: Second|Raichu|100/100",
            "|move|p2a: Rival|Tackle|p1a: Second",
            "|-damage|p1a: Second|0 fnt",
            "|faint|p1a: Second",
            "|switch|p1a: Third|Raichu|100/100",
            "|turn|2",
            "|switch|p1a: First|Pikachu|100/100",
            "|move|p2a: Rival|Tackle|p1a: First",
        )
    )
    complete_path = tmp_path / "complete.log"
    complete_path.write_text(protocol + "\n|win|Alice\n")
    incomplete_path = tmp_path / "incomplete.log"
    incomplete_path.write_text(protocol)

    report = uniform_kl_screen.protocol_switch_report([complete_path, incomplete_path])

    assert report["completed_games"] == 1
    assert report["skipped_logs"] == 1
    assert report["voluntary_switch_count"] == 1
    assert report["observed_move_or_switch_count"] == 4
    assert report["voluntary_switch_fraction"] == pytest.approx(0.25)
    assert report["ceiling_fraction"] == pytest.approx(0.161553269441742)


def test_protocol_switch_report_empty_is_unmeasured(monkeypatch):
    monkeypatch.setattr(uniform_kl_screen.battle_stats, "load_moves", dict)

    report = uniform_kl_screen.protocol_switch_report([])

    assert report["observed_move_or_switch_count"] == 0
    assert report["voluntary_switch_fraction"] is None
