from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from src.config import load_experiment_config
from src.data.data_workspace import DataWorkspace


@pytest.mark.parametrize("pretrained", [True, False])
def test_build_model_uses_model_config(monkeypatch, pretrained):
    torch = pytest.importorskip("torch")

    import src.modules.utils as module_utils
    from src.training.builders import build_model

    class FakeFeatureInfo:
        @staticmethod
        def reduction():
            return [4, 8, 16, 32]

        @staticmethod
        def channels():
            return [8, 16, 32, 64]

    class FakeEncoder(torch.nn.Module):
        feature_info = FakeFeatureInfo()

        def forward(self, image):
            batch, _, height, width = image.shape
            return [
                image.new_zeros(batch, 8, height // 4, width // 4),
                image.new_zeros(batch, 16, height // 8, width // 8),
                image.new_zeros(batch, 32, height // 16, width // 16),
                image.new_zeros(batch, 64, height // 32, width // 32),
            ]

    def create_encoder(*args, **kwargs):
        assert kwargs["pretrained"] is pretrained
        return FakeEncoder()

    monkeypatch.setattr(module_utils.timm, "create_model", create_encoder)

    config = load_experiment_config("configs/baseline.yaml")
    model = build_model(config.model, pretrained=pretrained)

    assert model.aux_weight == config.loss.aux_weight
    assert model.decoder.out_channels == FakeFeatureInfo.channels()[0]


def test_build_model_selects_independent_pvt_dgforce_model(monkeypatch):
    from src.config import ModelConfig
    from src.training import builders

    captured = {}

    class FakeDGForce:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.forensic_fusion = type('Fusion', (), {'branch': type('Branch', (), {
                'artifact': type('Artifact', (), {
                    'dc_layer0_dil': [type('Layer', (), {})()],
                    'dc_layer1_tail': [type('Layer', (), {})()],
                })()
            })()})()

    monkeypatch.setattr('src.modules.pvt_dgforce_segmenter.PVTDGForceSegmenter', FakeDGForce)
    monkeypatch.setattr('src.modules.sync_batchnorm.SynchronizedBatchNorm.apply', lambda model: model)
    config = ModelConfig(architecture='pvt_dgforce', jpeg_pretrained=None,
                         dgforce_reduction=8, dgforce_attention_width=64,
                         dgforce_attention_heads=4)

    model = builders.build_model(config, aux_weight=0.0, pretrained=False)

    assert isinstance(model, FakeDGForce)
    assert captured['encoder'] == 'pvt_v2_b2'
    assert captured['reduction'] == 8
    assert captured['attention_width'] == 64
    assert captured['attention_heads'] == 4


def test_build_loaders_uses_train_config():
    from src.training.builders import build_datasets, build_loaders

    config = load_experiment_config("configs/baseline.yaml")
    train_config = replace(
        config.train,
        device="cpu",
        workers=0,
        samples_per_epoch=6,
        batch_size=2,
    )
    workspace = DataWorkspace(Path("D:/data"))
    df = pd.DataFrame(
        [
            {
                "chng_img_path": "stage1/train/img/changed.jpg",
                "gt_path": "stage1/train/mask/changed.png",
                "is_negative": False,
            },
            {
                "chng_img_path": "stage1/train/img/negative.jpg",
                "gt_path": "stage1/train/mask/negative.png",
                "is_negative": True,
            },
        ]
    )

    train_ds, val_ds = build_datasets(config, workspace, df, df.iloc[:1])
    train_loader, val_loader = build_loaders(train_config, train_ds, val_ds)

    assert train_loader.batch_size == 2
    assert train_loader.sampler.num_samples == 6
    assert val_loader.batch_size == 2


def test_build_optimizer_groups_named_parameters():
    torch = pytest.importorskip("torch")

    from src.training.builders import build_optimizer

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(3, 3, 1, bias=False),
                torch.nn.BatchNorm2d(3),
            )
            self.forensic_fusion = torch.nn.Sequential(
                torch.nn.Conv2d(3, 3, 1, bias=False),
                torch.nn.BatchNorm2d(3),
            )
            self.decoder = torch.nn.Conv2d(3, 1, 1)

    config = load_experiment_config("configs/baseline.yaml")
    optimizer = build_optimizer(config.train, FakeModel())
    settings = {(group["lr"], group["weight_decay"]) for group in optimizer.param_groups}

    assert (config.train.encoder_lr, config.train.weight_decay) in settings
    assert (config.train.encoder_lr, 0.0) in settings
    assert (config.train.jpeg_lr, config.train.weight_decay) in settings
    assert (config.train.jpeg_lr, 0.0) in settings
    assert (config.train.head_lr, config.train.weight_decay) in settings
    assert (config.train.head_lr, 0.0) in settings


def test_build_scheduler_and_amp_context():
    torch = pytest.importorskip("torch")

    from src.training.builders import build_amp, build_scheduler

    config = load_experiment_config("configs/baseline.yaml")
    train_config = replace(config.train, device="cpu", amp="bf16")
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=train_config.head_lr)

    scheduler = build_scheduler(train_config, optimizer, steps_per_epoch=10)
    amp = build_amp(train_config)

    assert scheduler.lr_lambdas[0](0) > 0.0
    assert scheduler.lr_lambdas[0](10_000) == train_config.min_lr_factor
    assert amp.device.type == "cpu"
    assert amp.dtype is torch.bfloat16
    assert amp.enabled is False
    assert amp.scaler_enabled is False
