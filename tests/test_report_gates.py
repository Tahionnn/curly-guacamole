import numpy as np
import torch


def test_gate_report_measures_magnitudes_and_ignores_other_weights():
    from src.tools.report_gates import GateReport

    report = GateReport({'branch.channel_gate': torch.tensor([-.2, .4]),
                         'other.gamma': torch.zeros(2), 'projection.weight': torch.ones(2)})
    rows = report.rows()
    assert [row[0] for row in rows] == ['branch.channel_gate', 'other.gamma']
    np.testing.assert_allclose(rows[0][1:], [.3, .4])
    assert rows[1][1:] == (0., 0.)
