import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.training.metric import AICAccumulator


@pytest.fixture
def saved_run(tmp_path):
    run = tmp_path / 'run'
    (run / 'oof').mkdir(parents=True)
    (run / 'ckpt').mkdir()
    # Retuning must not deserialize or rewrite model weights.
    (run / 'ckpt/best.pt').write_bytes(b'unchanged checkpoint')
    (run / 'config.yaml').write_text(yaml.safe_dump({'n_bins': 100}), encoding='utf-8')
    acc = AICAccumulator(n_bins=100)
    probs = np.full((2, 20, 20), .1)
    gt = np.zeros_like(probs, dtype=bool)
    gt[0, :10] = True
    probs[0, :10] = .6
    probs[:, 0, 0] = .95
    probs[1, :10] = .6
    acc.update(probs, gt, np.array([.2, .2]))
    acc.save(run / 'oof/val.npz')
    old = acc.evaluate(.5, .5).as_dict()
    (run / 'summary.json').write_text(json.dumps({'best': old, 'val_best': old,
                                                 'best_aic': old['aic'], 'training_complete': True}), encoding='utf-8')
    return run


def test_retune_write_freezes_selection_without_changing_checkpoint(saved_run):
    from src.tools.retune import RunRetuner
    from src.training.selection import RetunedSelection

    retuner = RunRetuner(saved_run, small_mask_weight=1.)
    ranked = retuner.sweep([.5], [.5], [0.], [0., .01])
    assert ranked[0].area_cap == .01
    assert ranked[0].dice_pos == pytest.approx(2 / 201)
    before = (saved_run / 'summary.json').read_bytes()
    assert retuner.summary['best']['area_cap'] == 0.
    assert (saved_run / 'summary.json').read_bytes() == before
    retuner.save(ranked[0])
    summary = json.loads((saved_run / 'summary.json').read_text())
    assert summary['best'] == summary['val_best'] == ranked[0].as_dict()
    assert summary['best_aic'] == ranked[0].aic
    assert summary['retuned']['previous']['area_cap'] == 0.
    assert (saved_run / 'ckpt/best.pt').read_bytes() == b'unchanged checkpoint'
    assert RetunedSelection.verify(summary, {'n_bins': 100}, saved_run / 'ckpt/best.pt', check_histograms=True)
    # A second selection is possible before holdout and retains the prior point.
    another = RunRetuner(saved_run)
    another.save(another.sweep([.5], [.5], [0.], [0.])[0])
    summary = json.loads((saved_run / 'summary.json').read_text())
    assert summary['retuned']['previous']['area_cap'] == .01


@pytest.mark.parametrize('change', ['checkpoint', 'histograms', 'point', 'bins'])
def test_frozen_retune_rejects_changed_inputs(saved_run, change):
    from src.tools.retune import RunRetuner
    from src.training.selection import RetunedSelection

    retuner = RunRetuner(saved_run)
    retuner.save(retuner.sweep([.5], [.5], [0.], [.01])[0])
    summary = json.loads((saved_run / 'summary.json').read_text())
    snapshot = {'n_bins': 100}
    if change == 'checkpoint':
        (saved_run / 'ckpt/best.pt').write_bytes(b'different weights')
    elif change == 'histograms':
        (saved_run / 'oof/val.npz').write_bytes(b'different predictions')
    elif change == 'point':
        summary['best']['cls_threshold'] = .9
    else:
        snapshot['n_bins'] = 256
    with pytest.raises(ValueError, match='[Rr]etun'):
        RetunedSelection.verify(summary, snapshot, saved_run / 'ckpt/best.pt', check_histograms=True)


@pytest.mark.parametrize('state', ['claimed', 'evaluated', 'incomplete'])
def test_retune_cannot_write_after_holdout_or_during_training(saved_run, state):
    from src.tools.retune import RunRetuner

    summary_path = saved_run / 'summary.json'
    summary = json.loads(summary_path.read_text())
    if state == 'claimed':
        (saved_run / 'holdout_claim.json').write_text('{}')
    elif state == 'evaluated':
        summary['holdout_evaluated'] = True
    else:
        summary['training_complete'] = False
    summary_path.write_text(json.dumps(summary))
    retuner = RunRetuner(saved_run)
    before = summary_path.read_bytes()
    with pytest.raises(ValueError, match='holdout|[Cc]omplete'):
        retuner.save(retuner.sweep([.5], [.5], [0.], [.01])[0])
    assert summary_path.read_bytes() == before


@pytest.mark.parametrize('weight', [0., -1., float('nan'), float('inf')])
def test_retune_rejects_invalid_selection_weight(saved_run, weight):
    from src.tools.retune import RunRetuner

    with pytest.raises(ValueError, match='small_mask_weight'):
        RunRetuner(saved_run, small_mask_weight=weight)


def test_retune_cli_is_read_only_and_does_not_import_torch(saved_run):
    before = (saved_run / 'summary.json').read_bytes()
    script = (
        'import runpy, sys; '
        'sys.argv = ["retune", "--run", sys.argv[1], "--mask-thresholds", ".5", '
        '"--cls-thresholds", ".5", "--small-mask-weight", "1"]; '
        'runpy.run_module("src.tools.retune", run_name="__main__"); '
        'assert "torch" not in sys.modules'
    )
    result = subprocess.run([sys.executable, '-c', script, str(saved_run)],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--n-bins 100' in result.stdout
    assert (saved_run / 'summary.json').read_bytes() == before
