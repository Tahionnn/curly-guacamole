from dataclasses import replace

import pytest

from src.config import load_experiment_config


def _cpu_config(**train_overrides):
    config = load_experiment_config("configs/baseline.yaml")
    train = replace(
        config.train,
        device="cpu",
        workers=0,
        amp="off",
        grad_clip=0.0,
        **train_overrides,
    )
    return replace(
        config,
        train=train,
        loss=replace(config.loss, aux_weight=0.0),
        eval=replace(
            config.eval,
            n_bins=16,
            mask_thresholds=(0.5,),
            cls_thresholds=(0.5,),
            min_areas=(0.0,),
        ),
    )


class CountingEma:
    def __init__(self):
        self.updates = 0

    def update_parameters(self, model) -> None:
        self.updates += 1


class CountingScheduler:
    def __init__(self):
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


def test_train_one_epoch_flushes_last_accumulation_block():
    torch = pytest.importorskip("torch")

    from src.training.builders import build_amp
    from src.training.engine import train_one_epoch

    class TinySegmenter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Conv2d(3, 1, kernel_size=1)

        def forward(self, image, fmap=None):
            logits = self.head(image)
            return {
                "logits": logits,
                "cls_logits": logits.mean(dim=(2, 3)),
            }

    config = _cpu_config(grad_accum_steps=2)
    model = TinySegmenter()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = CountingScheduler()
    ema = CountingEma()
    amp = build_amp(config.train, torch.device("cpu"))

    loader = [
        {
            "image": torch.zeros(2, 3, 8, 8),
            "fmap": torch.zeros(2, 12, 1, 1),
            "mask": torch.zeros(2, 1, 8, 8),
            "label": torch.zeros(2, 1),
        }
        for _ in range(3)
    ]

    result = train_one_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=amp.scaler(),
        ema=ema,
        amp=amp,
        config=config,
        device=torch.device("cpu"),
    )

    assert result.seen == 6
    assert result.loss > 0.0
    assert result.loss_components["total"] == pytest.approx(result.loss)
    assert "dice_neg" in result.loss_components
    assert ema.updates == 2
    assert scheduler.steps == 2


def test_validate_accumulates_aic_histograms():
    torch = pytest.importorskip("torch")

    from src.training.builders import build_amp
    from src.training.engine import validate

    class ImageAsPrediction(torch.nn.Module):
        def forward(self, image, fmap=None):
            logits = image[:, :1] * 20.0 - 10.0
            return {
                "logits": logits,
                "cls_logits": logits.amax(dim=(2, 3)),
            }

    config = _cpu_config()
    amp = build_amp(config.train, torch.device("cpu"))

    image = torch.zeros(2, 3, 4, 4)
    image[0, :, :2, :2] = 1.0
    mask = torch.zeros(2, 1, 4, 4)
    mask[0, :, :2, :2] = 1.0

    validation = validate(
        ImageAsPrediction(),
        [
            {
                "image": image,
                "fmap": torch.zeros(2, 12, 1, 1),
                "mask": mask,
                "original_mask": [m[0].bool() for m in mask],
            }
        ],
        amp,
        config,
        torch.device("cpu"),
    )
    acc = validation.accumulator
    result = validation.tuned
    assert validation.operating_point == (0.5, 0.5, 0.0, 0.0)

    assert len(acc) == 2
    assert result.n_pos == 1
    assert result.n_neg == 1
    assert result.aic > 0.99
    assert validation.loss_components["total"] > 0
    assert validation.fixed.mask_threshold == 0.5
    assert validation.fixed.cls_threshold == 0.0


def test_resume_state_starts_after_saved_epoch(tmp_path):
    torch = pytest.importorskip("torch")

    from src.training.engine import ExperimentRunner

    class StateObject:
        def __init__(self):
            self.loaded = None

        def load_state_dict(self, state) -> None:
            self.loaded = state

    class FakeRun:
        def __init__(self, run_dir):
            self.dir = run_dir
            self.summary = {"best_aic": 0.5}
            self.messages = []

        def load_state(self, name, map_location):
            return {
                "model": {"model": True},
                "ema": {"ema": True},
                "optimizer": {"optimizer": True},
                "scheduler": {"scheduler": True},
                "epoch": 2,
                "samples": 17,
                "best_aic": 0.4,
            }

        def info(self, message) -> None:
            self.messages.append(message)

    config = _cpu_config(resume=True, samples_per_epoch=5)
    runner = ExperimentRunner(config)
    run_dir = tmp_path / "run"
    (run_dir / "ckpt").mkdir(parents=True)
    (run_dir / "ckpt" / "last.pt").touch()

    model = StateObject()
    optimizer = StateObject()
    scheduler = StateObject()
    ema = type("Ema", (), {"module": StateObject(), "n_averaged": torch.tensor(0)})()
    scaler = StateObject()

    state = runner._resume_if_needed(
        run=FakeRun(run_dir),
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )

    assert state.start_epoch == 3
    assert state.seen_total == 17
    assert state.best_aic == 0.5
    assert model.loaded == {"model": True}
    assert ema.module.loaded == {"ema": True}



@pytest.mark.parametrize('pixel_loss,boundary', [('bce', 0.), ('focal', -.05)])
def test_epoch_logs_loss_components_and_fixed_validation(tmp_path, pixel_loss, boundary):
    import json
    import time
    from types import SimpleNamespace

    import numpy as np

    from src.training.engine import EpochTrainResult, ExperimentRunner
    from src.training.metric import AICAccumulator
    from src.training.runs import Run
    from src.training.validation import ValidationResult

    config = _cpu_config()
    runner = ExperimentRunner(config)
    acc = AICAccumulator(n_bins=16)
    acc.update(np.array([[[.9]], [[.1]]]), np.array([[[1]], [[0]]]))
    score = acc.evaluate(.5)
    components = {"total": .6 + boundary, pixel_loss: .1, "dice": .2, "cls": .3}
    if boundary:
        components['boundary'] = boundary
    validation = ValidationResult(acc, score, loss_components=components, fixed=score)
    result = EpochTrainResult(.9, 0, 2, .5, {"total": .9, "bce": .2, "dice": .4, "cls": .3, "dice_pos": .5})
    with Run.create(tmp_path, "logging", tensorboard=False) as run:
        runner._log_epoch(run=run, epoch=0, model=SimpleNamespace(forensic_gate_stats=lambda: {"max_abs": 0},
                                               disentangle_gate_stats=lambda: {"max_abs": 0}),
                          train_result=result, tuned=score, seen_total=2, started=time.time(), steps_per_epoch=1,
                          optimizer=SimpleNamespace(param_groups=[{"lr": .001}]), validation=validation)
    row = json.loads(run.jsonl_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["train/loss"] == .9
    assert row["train/loss_dice_pos"] == .5
    assert row["val/loss_main"] == pytest.approx(.3 + boundary)
    assert row["val/loss_total"] == .6 + boundary
    assert row["val/aic_fixed"] == score.aic
    assert "val/loss_main" in run.csv_path.read_text(encoding="utf-8")
