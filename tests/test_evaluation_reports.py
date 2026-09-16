import numpy as np
import pandas as pd
import pytest

from src.inference.predict import ThresholdConfig
from src.training.metric import AICAccumulator


def test_report_separates_original_negatives_and_does_not_invent_negative_only_aic():
    from src.eval.diagnostics import EvaluationReport

    acc = AICAccumulator(n_bins=16)
    acc.update(np.array([[[.9, .1]], [[.9, .1]], [[.9, .1]]]),
               np.array([[[1., 0.]], [[0., 0.]], [[0., 0.]]]), np.array([1., .1, 1.]))
    rows = pd.DataFrame(dict(domain=['a'] * 3, target_kind=['provided', 'provided', 'original_zero'],
                             q_kind=['unit', 'unit', 'nonunit'], chng_img_path=['a', 'b', 'c']))
    report = EvaluationReport(acc, rows, ThresholdConfig(.5, .5, 0.))
    summary = report.summary()
    assert summary['provided']['false_positives'] == 0
    assert summary['originals']['false_positives'] == 1
    assert summary['originals']['aic'] is None
    assert summary['originals']['dice_pos'] is None
    assert summary['provided']['aic'] > .99
    assert report.per_image().false_positive_no_gate.tolist() == [False, True, True]
    with pytest.raises(ValueError, match='rows'):
        EvaluationReport(acc, rows.iloc[:2], ThresholdConfig())


def test_historical_originals_exclude_any_connection_to_training():
    from src.eval.checkpoints import historical_originals

    metadata = pd.DataFrame(dict(group_id=['a', 'b', 'c'], chng_img_path=['a.jpg', 'b.jpg', 'c.jpg'],
                                  orgl_img_path=['s1', 's2', 's3']))
    pairs = pd.DataFrame(dict(group_id=['a', 'b', 'c'], orgl_img_path=['s1', 's2', 's3'],
                             pixel_hash=['same', 'same', 'other']))
    originals = pd.DataFrame(dict(group_id=['a', 'c'], chng_img_path=['s1', 's3']))
    selected, excluded = historical_originals(metadata, metadata.iloc[[0, 2]], originals, pairs)
    assert selected.chng_img_path.tolist() == ['s3']
    assert excluded.chng_img_path.tolist() == ['s1']


def test_holdout_claim_cannot_be_reopened(tmp_path):
    from src.eval.checkpoints import claim_holdout

    claim_holdout(tmp_path, {'checkpoint': 'abc'})
    with pytest.raises(FileExistsError):
        claim_holdout(tmp_path, {'checkpoint': 'def'})


@pytest.mark.parametrize('batch_size,workers', [(0, 0), (2, -1)])
def test_invalid_evaluation_arguments_fail_before_opening_run(tmp_path, batch_size, workers):
    from src.eval.checkpoints import CheckpointEvaluator

    with pytest.raises(ValueError, match='batch_size|workers'):
        CheckpointEvaluator(tmp_path / 'missing', device='cpu', batch_size=batch_size, workers=workers)


@pytest.mark.parametrize('retune', [False, True])
def test_holdout_pipeline_loads_frozen_checkpoint_and_writes_separate_metrics(tmp_path, monkeypatch, retune):
    from dataclasses import replace

    import torch
    from PIL import Image

    from src.config import load_experiment_config
    from src.eval.checkpoints import CheckpointEvaluator
    from src.eval.protocol import EvaluationProtocol
    from src.training.runs import Run

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(-10.))

        def forward(self, image, **kwargs):
            return {'logits': image[:, :1] * 0 + self.bias,
                    'cls_logits': self.bias.expand(len(image), 1)}

    root = tmp_path / 'data' / 'train_stage1'
    root.mkdir(parents=True)
    records = []
    for i in range(60):
        Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(root / f'{i}.jpg')
        Image.fromarray(np.full((16, 16), 255 if i % 2 else 0, np.uint8)).save(root / f'{i}.png')
        records.append(dict(chng_img_path=f'{i}.jpg', gt_path=f'{i}.png', orgl_img_path=None,
                            group_id=str(i), domain='plain', generator='none', mask_area=1. if i % 2 else 0.,
                            is_negative=not bool(i % 2), broken=False, stem=str(i)))
    protocol = EvaluationProtocol.create(tmp_path / 'p', pd.DataFrame(records), pd.DataFrame(), pd.DataFrame())
    config = load_experiment_config('configs/baseline.yaml')
    config = replace(config, paths=replace(config.paths, data_path=root.parent),
                     dataset=replace(config.dataset, image_size=16, protocol_path=str(protocol.path)),
                     eval=replace(config.eval, n_bins=100),
                     train=replace(config.train, device='cpu', amp='off', workers=0))
    snapshot = dict(config.to_flat_dict(), **protocol.provenance())
    run = Run.create(tmp_path, 'run', tensorboard=False)
    run.save_snapshot(snapshot)
    point = dict(mask_threshold=.75, cls_threshold=.5, min_area=0.)
    run.save_summary(dict(best=point, training_complete=True))
    run.save_state(dict(model=TinyModel().state_dict(), cfg=snapshot, operating_point=point), 'best.pt')
    protocol.rows('train').to_parquet(run.dir / 'training_rows.parquet', index=False)
    protocol.rows('development').to_parquet(run.dir / 'development_rows.parquet', index=False)
    if retune:
        from src.tools.retune import RunRetuner

        dev = protocol.rows('development')
        acc = AICAccumulator(n_bins=100)
        gt = np.array([np.full((16, 16), not row.is_negative) for row in dev.itertuples()])
        acc.update(np.full(gt.shape, 1 / (1 + np.exp(10))), gt, np.zeros(len(dev)))
        acc.save(run.dir / 'oof/val.npz')
        retuner = RunRetuner(run.dir)
        result = retuner.sweep([.37], [.5], [0.], [.01])[0]
        retuner.save(result)
        point = result.as_dict()
    monkeypatch.setattr('src.eval.checkpoints.build_model', lambda *args, **kwargs: TinyModel())
    monkeypatch.setenv('AIIJC_DATA_PATH', str(tmp_path / 'data'))
    evaluator = CheckpointEvaluator(run.dir, device='cpu', batch_size=2, workers=0)
    result = evaluator.holdout()
    assert result['provided']['n_pos'] > 0 and result['provided']['n_neg'] > 0
    assert result['provided']['false_positives'] == 0
    assert result['provided']['dice_pos'] == 0
    assert (run.dir / 'holdout/per_image.parquet').exists()
    assert run.summary['best'] == point
    assert evaluator.thresholds.n_bins == 100
    with pytest.raises(FileExistsError):
        evaluator.holdout()
    run.close()
