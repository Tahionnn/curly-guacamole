import numpy as np  # noqa: F401 -- initialize NumPy's runtime before torch on Windows.
import pytest
import torch

from src.config import load_experiment_config
from src.modules.jpeg_branch import JPEGArtifactModule


def test_load_pretrained_stem_strictly(tmp_path):
    source = JPEGArtifactModule()
    state = source.state_dict()
    state['dc_layer0_dil.0.weight'].fill_(0.125)
    path = tmp_path / 'weights.pth'
    torch.save({'state_dict': {f'module.{k}': v for k, v in state.items()}}, path)
    target = JPEGArtifactModule()
    target.load_pretrained(path)
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, state[key])
    del state['dc_layer1_tail.0.weight']
    torch.save({'state_dict': state}, path)
    with pytest.raises(RuntimeError, match='Missing key'):
        target.load_pretrained(path)


def test_pretrained_recipe_and_builder_gate(monkeypatch):
    from dataclasses import replace

    import src.modules.segmenter as segmenter
    from src.modules.sync_batchnorm import SynchronizedBatchNorm
    from src.training.builders import build_model

    config = load_experiment_config('configs/baseline.yaml')
    assert config.model.jpeg_pretrained == 'DCT_djpeg.pth'
    calls = []

    class FakeModel:
        def __init__(self, **kwargs):
            from types import SimpleNamespace
            self.forensic_fusion = SimpleNamespace(branch=SimpleNamespace(artifact=self))
            self.dc_layer0_dil = [SimpleNamespace()]
            self.dc_layer1_tail = [SimpleNamespace()]

        def load_pretrained(self, path):
            calls.append(path)

    monkeypatch.setattr(segmenter, 'Segmenter', FakeModel)
    monkeypatch.setattr(SynchronizedBatchNorm, 'apply', lambda model: model)
    build_model(config.model, pretrained=False)
    assert not calls
    build_model(config.model, pretrained=True)
    assert len(calls) == 1 and calls[0].is_absolute()
    build_model(replace(config.model, jpeg_pretrained=None), pretrained=True)
    assert len(calls) == 1


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_load_legacy_checkpoint_with_numpy_metric(tmp_path, dtype):
    source = JPEGArtifactModule()
    path = tmp_path / 'legacy.pth'
    torch.save({'state_dict': source.state_dict(), 'best_p_mIoU': dtype(.75),
                'epoch': 100, 'optimizer': {}}, path, _use_new_zipfile_serialization=False)
    before = set(torch.serialization.get_safe_globals())
    target = JPEGArtifactModule()
    target.load_pretrained(path)
    assert set(torch.serialization.get_safe_globals()) == before
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, source.state_dict()[key], rtol=0, atol=0)
