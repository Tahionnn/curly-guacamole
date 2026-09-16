"""The capped classifier gate, and the one rule shared by sweep/report/inference."""

import numpy as np
import pytest

from src.training.metric import FP_AREA_THRESHOLD, AICAccumulator, operating_bins

# src.inference pulls in torch; the metric-side tests must not need a GPU stack.


def _accumulator(n_bins=64, seed=0):
    rng = np.random.default_rng(seed)
    acc = AICAccumulator(n_bins=n_bins)
    for i in range(40):
        h, w = int(rng.integers(20, 40)), int(rng.integers(20, 40))
        gt = np.zeros((h, w), bool)
        if i % 3:
            gt[: rng.integers(1, h // 2), : rng.integers(1, w // 2)] = True
        probs = rng.beta(1.2, 9.0, (h, w))
        probs[gt] = rng.beta(7.0, 1.6, int(gt.sum()))
        cls = float(gt.any()) * .8 + .2 * rng.random()
        acc.update(np.clip(probs, 0, 1 - 1e-9)[None], gt[None].astype(np.float32), np.array([cls]))
    return acc


def test_area_cap_zero_reproduces_the_gate_that_zeroes_a_frame():
    """Default must be bit-identical to the pre-area_cap sweep, or old runs move."""
    acc = _accumulator()
    pred, inter, gt_sum, n_pixels, cls_prob = acc.tables()
    for k, cls_thr, min_area in [(8, 0.0, 0.0), (20, .5, 0.0), (20, .5, .02), (40, .9, .01)]:
        bins, blank = operating_bins(pred, n_pixels, cls_prob, k, cls_thr, min_area, 0.0)
        area = pred[:, k] / n_pixels
        assert (bins == k).all()
        np.testing.assert_array_equal(blank, (cls_prob < cls_thr) | (area < min_area))


def test_a_gated_frame_can_never_be_a_false_alarm_under_a_cap():
    acc = _accumulator()
    pred, _, _, n_pixels, cls_prob = acc.tables()
    rows = np.arange(pred.shape[0])
    for area_cap in (.004, .008, FP_AREA_THRESHOLD):
        bins, blank = operating_bins(pred, n_pixels, cls_prob, 5, .9, 0.0, area_cap)
        gated = cls_prob < .9
        area = np.where(blank, 0.0, pred[rows, bins] / n_pixels)
        assert gated.any()
        assert (area[gated] < area_cap).all()
        assert not (area[gated] >= FP_AREA_THRESHOLD).any()


@pytest.mark.parametrize('min_area', [0.0, .002, .005, FP_AREA_THRESHOLD])
def test_min_area_at_or_below_the_alarm_threshold_cannot_change_fpr(min_area):
    """Documents why min_areas stays [0.0]: below 1% it only costs Dice.

    A negative frame is a false alarm from 1% of the frame, so blanking anything
    smaller removes only frames that were already not false alarms.
    """
    acc = _accumulator()
    grid, cls = [k / acc.n_bins for k in range(4, acc.n_bins, 4)], [0.0, .5, .9]
    for threshold in grid:
        for cls_threshold in cls:
            base = acc.evaluate(threshold, cls_threshold, 0.)
            tried = acc.evaluate(threshold, cls_threshold, min_area)
            assert tried.fpr_neg == base.fpr_neg
            assert tried.dice_pos <= base.dice_pos + 1e-12
            assert tried.aic <= base.aic + 1e-12


def test_adding_caps_to_the_grid_can_never_lose():
    """area_cap=0 stays in the grid, so the search can only find something better."""
    acc = _accumulator()
    grid, cls = [k / acc.n_bins for k in range(4, acc.n_bins, 2)], [round(x, 2) for x in np.arange(0, .99, .05)]
    zeroing = acc.best(grid, cls, [0.0], [0.0])
    capped = acc.best(grid, cls, [0.0], [0.0, .008, FP_AREA_THRESHOLD])
    assert capped.aic >= zeroing.aic
    assert capped.as_dict()['area_cap'] == capped.area_cap


def test_cap_keeps_the_dice_of_a_positive_the_classifier_wrongly_doubts():
    """The whole point: gating a positive costs all of its Dice, capping does not.

    The prediction needs a graded core, not one flat level -- raising the
    threshold on a uniform blob takes its area from everything to nothing, and
    there is nothing left for the cap to keep.
    """
    n_bins, side = 64, 40
    pixels = side * side                                  # 1600; the 1% alarm is 16 px
    acc = AICAccumulator(n_bins=n_bins)
    gt = np.zeros((side, side), bool)
    gt[: side // 2] = True                                # 800 px of ground truth

    for doubted in (False, True):
        probs = np.where(gt, .6, .1)
        probs.reshape(-1)[:10] = .99                      # a 10 px high-confidence core
        acc.update(probs[None], gt[None].astype(np.float32), np.array([.1 if doubted else .9]))
    # A negative the classifier also doubts, predicted far above the alarm area.
    acc.update(np.full((1, side, side), .9), np.zeros((1, side, side), np.float32), np.array([.1]))

    zeroing = acc.evaluate(.5, .5, 0.0, 0.0)
    capped = acc.evaluate(.5, .5, 0.0, FP_AREA_THRESHOLD)

    # Both silence the doubted negative: that is not where they differ.
    assert zeroing.fpr_neg == 0.0 and capped.fpr_neg == 0.0
    # The confident positive scores 1.0 either way; the doubted one is the test.
    # Zeroing throws it away, the cap keeps its 10 px core inside the truth.
    assert zeroing.dice_pos == pytest.approx(.5, abs=1e-6)
    assert capped.dice_pos == pytest.approx((1.0 + 2 * 10 / (10 + 800)) / 2, abs=1e-4)
    assert capped.dice_pos > zeroing.dice_pos
    assert capped.aic > zeroing.aic
    assert 10 / pixels < FP_AREA_THRESHOLD                # the kept core is not an alarm


def test_threshold_config_requires_a_threshold_validation_could_have_chosen():
    pytest.importorskip('torch')
    from src.inference.predict import ThresholdConfig

    ThresholdConfig(0.5, 0.0, 0.0, area_cap=FP_AREA_THRESHOLD)
    assert ThresholdConfig(108 / 256).bin_index == 108
    # Clamped: the grid has no bin for 1.0, so the top bin is the honest reading.
    assert ThresholdConfig(1.0).bin_index == 255
    with pytest.raises(ValueError, match='histogram grid'):
        ThresholdConfig(0.3)
    with pytest.raises(ValueError, match='area_cap'):
        ThresholdConfig(0.5, area_cap=1.5)


def test_inference_masks_score_exactly_what_the_sweep_promised():
    """Selection and submission must read one rule; a private copy would drift."""
    torch = pytest.importorskip('torch')
    from src.inference.predict import Predictor, ThresholdConfig
    from src.training.builders import AmpContext
    from src.training.metric import EPS, harmonic_aic

    rng = np.random.default_rng(3)
    frames, n_bins = [], 64
    for i in range(30):
        h, w = int(rng.integers(20, 40)), int(rng.integers(20, 40))
        gt = np.zeros((h, w), bool)
        if i % 3:
            gt[: rng.integers(1, h // 2), : rng.integers(1, w // 2)] = True
        probs = rng.beta(1.2, 9.0, (h, w))
        probs[gt] = rng.beta(7.0, 1.6, int(gt.sum()))
        frames.append((np.clip(probs, 0, 1 - 1e-9), gt, float(gt.any()) * .8 + .2 * rng.random()))

    acc = AICAccumulator(n_bins=n_bins)
    for probs, gt, cls in frames:
        acc.update(probs[None], gt[None].astype(np.float32), np.array([cls]))

    amp = AmpContext(torch.device('cpu'), torch.float32, False, False)
    for k, cls_thr, min_area, area_cap in [(20, 0.0, 0.0, 0.0), (20, .85, 0.0, 0.0),
                                           (20, .85, 0.0, .0095), (32, .85, .005, .008)]:
        swept = acc.evaluate(k / n_bins, cls_thr, min_area, area_cap)
        predictor = Predictor(torch.nn.Identity(),
                              ThresholdConfig(k / n_bins, cls_thr, min_area, area_cap, n_bins), amp)
        dices, alarms = [], []
        for probs, gt, cls in frames:
            mask = predictor.binary_mask(torch.tensor(probs)[None, None], cls, probs.shape) > 0
            if gt.any():
                dices.append(2 * (mask & gt).sum() / (mask.sum() + gt.sum() + EPS))
            else:
                alarms.append(float(mask.mean() >= FP_AREA_THRESHOLD))
        assert harmonic_aic(float(np.mean(dices)), float(np.mean(alarms))) == pytest.approx(swept.aic, abs=1e-9)


@pytest.mark.parametrize('threshold', [.29, .58])
def test_nonbinary_histogram_grid_round_trips_selected_threshold(threshold):
    import torch

    from src.inference.predict import Predictor, ThresholdConfig
    from src.training.builders import AmpContext

    probs = np.array([[[threshold - .001, threshold, threshold + .001]]])
    gt = np.ones_like(probs, dtype=bool)
    acc = AICAccumulator(n_bins=100)
    acc.update(probs, gt)
    selected = acc.evaluate(threshold)
    assert selected.mask_threshold == threshold
    cfg = ThresholdConfig(selected.mask_threshold, n_bins=100)
    assert cfg.bin_index == round(threshold * 100)
    predictor = Predictor(torch.nn.Identity(), cfg, AmpContext(torch.device('cpu'), torch.float32, False, False))
    mask = predictor.mask_from_probabilities(probs[0], 1.) > 0
    assert 2 * mask.sum() / (mask.sum() + 3 + 1e-6) == pytest.approx(selected.dice_pos)
