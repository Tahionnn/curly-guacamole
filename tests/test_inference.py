import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.config import load_experiment_config
from src.inference import Prediction, Predictor, SubmissionWriter, ThresholdConfig, create_submission
from src.training.builders import AmpContext


def test_restores_probabilities_before_threshold_and_applies_gates():
    predictor = Predictor(torch.nn.Identity(), ThresholdConfig(0.5, 0.5, 0.1),
                          AmpContext(torch.device("cpu"), torch.float16, False, False))
    probability = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
    result = predictor.binary_mask(probability, 0.5, (3, 5))
    assert result.shape == (3, 5)
    assert result.dtype == np.uint8
    assert np.all(result[:, :2] == 0)
    assert np.all(result[:, 2:] == 255)
    assert not predictor.binary_mask(probability, 0.49, (3, 5)).any()
    predictor.thresholds = ThresholdConfig(0.5, 0, 0.9)
    assert not predictor.binary_mask(probability, 1, (3, 5)).any()


@pytest.mark.parametrize("path", ["../outside.png", "/predictions/a.png", "C:/predictions/a.png",
                                 "predictions/../outside.png", "predictions/a.jpg", "predictions/a:b.png"])
def test_rejects_unsafe_prediction_paths(tmp_path, path):
    template = pd.DataFrame({"img_path": ["a.jpg"], "prediction_path": [path]})
    with pytest.raises(ValueError, match="prediction path"):
        SubmissionWriter(template, template[["img_path"]], tmp_path)


def test_template_alignment_and_prediction_completeness(tmp_path):
    template = pd.DataFrame({"img_path": ["a.jpg", "b.jpg"],
                             "prediction_path": ["predictions/a.png", "predictions/b.png"]})
    with pytest.raises(ValueError, match="match test.csv"):
        SubmissionWriter(template, pd.DataFrame({"img_path": ["a.jpg", "c.jpg"]}), tmp_path)
    writer = SubmissionWriter(template, template[["img_path"]], tmp_path)
    with pytest.raises(ValueError, match="missing"):
        writer.write([Prediction("a.jpg", np.zeros((3, 5), dtype=np.uint8))])
    assert not (tmp_path / "submission.csv").exists()


@pytest.mark.parametrize('checkpoint_name', ['best.pt', 'last.pt'])
def test_run_submission_prefers_ema_preserves_template_and_sizes(tmp_path, monkeypatch, checkpoint_name):
    import src.inference.submission as module
    from src.training.runs import Run

    run = Run.create(tmp_path, "run", tensorboard=False)
    config = load_experiment_config("configs/baseline.yaml")
    snapshot = config.to_flat_dict()
    snapshot.update(device="cpu", amp="off", workers=0, batch_size=2)
    run.save_snapshot(snapshot)
    run.save_summary({"best": {"mask_threshold": 0.5, "cls_threshold": 0.0, "min_area": 0.0}})
    torch.save({"model": {"bias": torch.tensor(-10.)}, "ema": {"bias": torch.tensor(10.)}},
               run.dir / "ckpt" / checkpoint_name)
    test_root = tmp_path / "data" / "test_stage1" / "test_stage1"
    test_root.mkdir(parents=True)
    pd.DataFrame({"img_path": ["a.jpg", "b.jpg"]}).to_csv(test_root / "test.csv", index=False)
    template = pd.DataFrame({"img_path": ["b.jpg", "a.jpg"],
                             "prediction_path": ["predictions/sub/b.png", "predictions/a.png"]})
    template.to_csv(test_root / "submission.csv", index=False)

    class DummyDataset:
        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {"image": torch.zeros(3, 8, 8), "fmap": torch.zeros(12, 1, 1),
                    "image_path": ["a.jpg", "b.jpg"][index],
                    "original_size": torch.tensor([(3, 5), (7, 4)][index])}

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.))

        def forward(self, image, **kwargs):
            return {"logits": self.bias.expand(len(image), 1, 8, 8),
                    "cls_logits": self.bias.expand(len(image), 1)}

    def model_factory(config, *, pretrained, aux_weight):
        assert pretrained is False
        return DummyModel()

    monkeypatch.setattr(module, "AIIJCDataset", DummyDataset)
    monkeypatch.setattr(module, "build_model", model_factory)
    csv = create_submission(run.dir, tmp_path / "output", data_path=tmp_path / "data",
                            checkpoint_name=checkpoint_name)
    pd.testing.assert_frame_equal(pd.read_csv(csv), template)
    for path, size in [("predictions/a.png", (5, 3)), ("predictions/sub/b.png", (4, 7))]:
        with Image.open(csv.parent / path) as mask:
            assert mask.size == size
            assert mask.mode == "L"
            assert np.all(np.asarray(mask) == 255)

    (run.dir / "summary.json").write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(ValueError, match="operating point"):
        create_submission(run.dir, tmp_path / "no_summary", data_path=tmp_path / "data")
    create_submission(run.dir, tmp_path / "explicit", data_path=tmp_path / "data",
                      checkpoint_name=checkpoint_name,
                      thresholds=ThresholdConfig(1.0))


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_threshold_config_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        ThresholdConfig(mask_threshold=value)
