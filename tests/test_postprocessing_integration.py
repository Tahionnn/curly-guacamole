from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.inference import Prediction, Predictor, SubmissionWriter, ThresholdConfig
from src.training.builders import AmpContext


class DeferredPool:
    """A slow executor: work finishes only when the consumer waits for it."""

    def __init__(self):
        self.pending = 0
        self.peak = 0

    def submit(self, function, *args):
        self.pending += 1
        self.peak = max(self.peak, self.pending)
        pool = self

        class Work:
            def result(self):
                try:
                    return function(*args)
                finally:
                    pool.pending -= 1

        return Work()


def test_writer_bounds_unfinished_pngs_and_waits_before_csv(tmp_path):
    names = [f'{i}.jpg' for i in range(9)]
    template = pd.DataFrame({'img_path': names, 'prediction_path': [f'predictions/{i}.png' for i in range(9)]})
    writer = SubmissionWriter(template, template[['img_path']], tmp_path)
    pool = DeferredPool()
    masks = [np.full((3, 4), 255 * (i % 2), np.uint8) for i in range(9)]
    csv = writer.write((Prediction(name, mask) for name, mask in zip(names, masks, strict=True)), pool=pool, queue_depth=2)
    assert pool.peak <= 2
    assert pool.pending == 0
    assert csv.exists()
    for i, mask in enumerate(masks):
        with Image.open(tmp_path / f'predictions/{i}.png') as saved:
            np.testing.assert_array_equal(np.asarray(saved), mask)


def test_writer_propagates_background_failure_without_csv(tmp_path, monkeypatch):
    template = pd.DataFrame({'img_path': ['a'], 'prediction_path': ['predictions/a.png']})
    writer = SubmissionWriter(template, template[['img_path']], tmp_path)

    def fail(*args):
        raise OSError('disk full')

    monkeypatch.setattr(writer, '_encode', fail, raising=False)
    with ThreadPoolExecutor(1) as pool, pytest.raises(OSError, match='disk full'):
        writer.write([Prediction('a', np.zeros((3, 4), np.uint8))], pool=pool)
    assert not (tmp_path / 'submission.csv').exists()


@pytest.mark.parametrize('queue_depth', [1, 2, 8])
def test_threaded_predictor_preserves_masks_order_and_queue_bound(queue_depth):
    class Model(torch.nn.Module):
        def forward(self, image):
            return {'logits': image[:, :1], 'cls_logits': torch.full((len(image), 1), -1.)}

    amp = AmpContext(torch.device('cpu'), torch.float32, False, False)
    predictor = Predictor(Model(), ThresholdConfig(.5, .5, area_cap=.01), amp)
    images = torch.full((5, 3, 20, 20), -2.)
    images[:, :, 0, 0] = 4.
    batch = dict(image=images, image_path=[str(i) for i in range(5)], original_size=[(20, 20)] * 5)
    expected = list(predictor.predict([batch]))
    pool = DeferredPool()
    actual = list(predictor.predict([batch], pool=pool, queue_depth=queue_depth))
    assert pool.peak <= queue_depth
    assert [item.image_path for item in actual] == [str(i) for i in range(5)]
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(left.mask, right.mask)
        assert np.count_nonzero(left.mask) == 1
    with ThreadPoolExecutor(2) as threads:
        assert len(list(predictor.predict([batch], pool=threads, queue_depth=queue_depth))) == 5


@pytest.mark.parametrize('snapshot', [{'n_bins': 100}, {'eval': {'n_bins': 100}}])
def test_evaluator_loads_cap_and_nondefault_histogram_grid(tmp_path, snapshot):
    from src.config import load_experiment_config
    from src.eval.checkpoints import CheckpointEvaluator
    from src.training.runs import Run

    run = Run.create(tmp_path, 'run', tensorboard=False)
    config = load_experiment_config('configs/baseline.yaml')
    saved = config.to_dict() if 'eval' in snapshot else config.to_flat_dict()
    saved.update(snapshot)
    run.save_snapshot(saved)
    run.save_summary({'best': dict(mask_threshold=.37, cls_threshold=.8, min_area=0., area_cap=.01)})
    torch.save({}, run.dir / 'ckpt/best.pt')
    evaluator = CheckpointEvaluator(run.dir, device='cpu')
    assert evaluator.thresholds.n_bins == 100
    assert evaluator.thresholds.bin_index == 37
    assert evaluator.thresholds.area_cap == .01


def test_report_and_frozen_validation_reproduce_capped_masks():
    from dataclasses import replace

    from src.config import load_experiment_config
    from src.eval.diagnostics import EvaluationReport
    from src.training.metric import AICAccumulator
    from src.training.validation import _score_validation

    acc = AICAccumulator(n_bins=256, small_mask_weight=1.6)
    probs = np.full((2, 20, 20), .6)
    probs[:, 0, 0] = .95
    gt = np.zeros_like(probs, dtype=bool)
    gt[0, 0, :10] = True
    acc.update(probs, gt, np.array([.1, .1]))
    thresholds = ThresholdConfig(.5, .5, area_cap=.01)
    report = EvaluationReport(acc, pd.DataFrame({'domain': ['a', 'b']}), thresholds)
    summary = report.summary()
    assert summary['combined']['false_positives'] == 0
    assert summary['combined']['dice_pos'] == pytest.approx(2 / 11)
    assert report.per_image().pred_fraction.tolist() == [.0025, .0025]
    config = load_experiment_config('configs/baseline.yaml')
    config = replace(config, eval=replace(config.eval, mask_thresholds=(.5,), cls_thresholds=(.5,),
                                          min_areas=(0.,), area_caps=(.01,)))
    for frozen in (None, thresholds):
        tuned, _ = _score_validation(acc, config, frozen)
        assert tuned.area_cap == .01
        assert tuned.aic == pytest.approx(summary['selection']['aic'])


@pytest.mark.parametrize('cls_threshold,min_area,cap,cls_probability,blank', [
    (0., 0., 0., .1, False),
    (.5, 0., 0., .1, True),
    (.5, 0., 0., .8, False),
    (.5, .5, .01, .8, False),
    (.5, .6, .01, .8, True),
])
def test_inactive_cap_skips_histogram_but_preserves_classifier_and_area_rules(
        monkeypatch, cls_threshold, min_area, cap, cls_probability, blank):
    predictor = Predictor(torch.nn.Identity(), ThresholdConfig(.5, cls_threshold, min_area, cap),
                          AmpContext(torch.device('cpu'), torch.float32, False, False))

    def histogram_was_not_needed(*args, **kwargs):
        pytest.fail('A frame without adaptive capping must not build a histogram')

    monkeypatch.setattr(np, 'bincount', histogram_was_not_needed)
    result = predictor.mask_from_probabilities(np.array([[.1, .6], [.7, .4]]), cls_probability)
    expected = np.zeros((2, 2), np.uint8) if blank else np.array([[0, 255], [255, 0]], np.uint8)
    np.testing.assert_array_equal(result, expected)


def test_predict_disables_shape_autotuning_only_inside_forward():
    class Model(torch.nn.Module):
        def forward(self, image):
            assert not torch.backends.cudnn.benchmark
            return {'logits': image[:, :1], 'cls_logits': torch.zeros(len(image), 1)}

    predictor = Predictor(Model(), ThresholdConfig(), AmpContext(torch.device('cpu'), torch.float32, False, False))
    batch = dict(image=torch.ones(1, 3, 2, 2), image_path=['a'], original_size=[(3, 4)])
    with torch.backends.cudnn.flags(benchmark=True):
        predictions = predictor.predict([batch])
        first = next(predictions)
        assert first.mask.shape == (3, 4)
        assert torch.backends.cudnn.benchmark
        assert list(predictions) == []
        assert torch.backends.cudnn.benchmark
