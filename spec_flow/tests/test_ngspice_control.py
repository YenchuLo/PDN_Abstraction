"""ngspice .control emit: KLU + single multi-node print."""

from __future__ import annotations

from pathlib import Path

from spice_emit import _analysis_print_ngspice


def test_analysis_print_ngspice_klu_single_print(tmp_path: Path):
    nodes = [("pad_0", "pad_0"), ("pad_1", "pad_1"), ("sink_0", "sink_0")]
    volt = tmp_path / "original.volt"
    lines = _analysis_print_ngspice(nodes, volt)

    assert lines[0] == ".control"
    assert lines[1] == "option klu"
    assert lines[2] == "op"
    assert lines[-1] == ".endc"

    print_lines = [ln for ln in lines if ln.startswith("print ")]
    assert len(print_lines) == 1
    assert print_lines[0] == (
        f"print v(pad_0) v(pad_1) v(sink_0) > {volt.as_posix()}"
    )
    assert ">>" not in "\n".join(lines)
