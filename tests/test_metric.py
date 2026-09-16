import numpy as np
import pytest

from src.training.metric import AICAccumulator, score_masks


def test_false_positive_area_includes_exactly_one_percent():
    gt = np.zeros((3, 10, 10), dtype=bool)
    gt[0, :2, :2] = True
    pred = gt.copy()
    pred[1, 0, 0] = True
    direct = score_masks(pred, gt)
    acc = AICAccumulator(n_bins=16)
    acc.update(pred.astype(float), gt)
    histogram = acc.evaluate()
    assert direct.fpr_neg == histogram.fpr_neg == 0.5
    assert direct.aic == pytest.approx(histogram.aic)


def test_tuned_threshold_matches_effective_histogram_boundary():
    probs = np.array([[[0.51, 0.57]], [[0.1, 0.1]]])
    gt = np.array([[[True, True]], [[False, False]]])
    acc = AICAccumulator(n_bins=16)
    acc.update(probs, gt)
    tuned = acc.best((0.55,))
    direct = score_masks(probs >= tuned.mask_threshold, gt)
    assert tuned.aic == pytest.approx(direct.aic)


def test_save_validation_result_preserves_tuned_operating_point(tmp_path):
    import pandas as pd

    from src.training.runs import Run
    from src.training.validation import ValidationResult

    acc = AICAccumulator(n_bins=16)
    acc.update(np.array([[[0.9]], [[0.1]]]), np.array([[[1]], [[0]]]))
    tuned = acc.best((0.75,), (0.25,), (0.0,))
    result = ValidationResult(accumulator=acc, tuned=tuned)
    run = Run.create(tmp_path, 'validation')
    try:
        run.save_eval(result, pd.DataFrame({'stem': ['positive', 'negative']}))
        assert run.operating_point() == result.operating_point == (0.75, 0.25, 0.0, 0.0)
        assert run.summary['best_aic'] == tuned.aic
        assert len(run.load_eval().acc) == 2
    finally:
        run.close()


def test_saved_operating_point_keeps_area_cap(tmp_path):
    from src.training.runs import Run
    from src.training.validation import ValidationResult

    acc = AICAccumulator()
    acc.update(np.full((1, 10, 10), .6), np.ones((1, 10, 10)), np.array([.1]))
    result = ValidationResult(acc, acc.evaluate(.5, .5, 0., .01))
    run = Run.create(tmp_path, 'capped', tensorboard=False)
    run.save_summary({'best': result.tuned.as_dict()})
    assert run.operating_point() == result.operating_point == (.5, .5, 0., .01)
