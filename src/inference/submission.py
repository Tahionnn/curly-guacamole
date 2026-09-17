"""Build competition submission.csv and predictions/ from a saved run."""

from collections import deque
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.config import ExperimentConfig, ModelConfig, SnapshotAdapter
from src.data.collation import ValidationCollator
from src.data.data_workspace import DataWorkspace
from src.data.dataset import AIIJCDataset
from src.inference.predict import Prediction, Predictor, ThresholdConfig
from src.progress import ConsoleProgress
from src.training.builders import AmpContext, build_model
from src.training.runs import Run
from src.training.selection import RetunedSelection


class SubmissionWriter:
    def __init__(self, template: pd.DataFrame, test_rows: pd.DataFrame, output_dir: str | Path):
        self.template = template.copy()
        self.output_dir = Path(output_dir).resolve()
        self._written: set[str] = set()
        for frame, columns in ((template, ("img_path", "prediction_path")), (test_rows, ("img_path",))):
            for column in columns:
                if column not in frame or frame[column].isna().any():
                    raise ValueError(f"missing values or column: {column}")
                if not frame[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
                    raise ValueError(f"{column} must contain nonempty strings")
                if frame[column].duplicated().any():
                    raise ValueError(f"duplicate values in {column}")
        if set(template.img_path) != set(test_rows.img_path):
            raise ValueError("template img_path values must match test.csv exactly")
        self.paths = {row.img_path: self._output_path(row.prediction_path) for row in template.itertuples()}
        # Detect aliases, including Windows case-insensitive filenames.
        if len({str(path).casefold() for path in self.paths.values()}) != len(self.paths):
            raise ValueError("prediction paths resolve to duplicate files")

    def _output_path(self, raw: str) -> Path:
        path = PurePosixPath(raw.replace("\\", "/"))
        if (PureWindowsPath(raw).drive or path.is_absolute() or ".." in path.parts
                or len(path.parts) < 2 or path.parts[0] != "predictions"
                or path.suffix.lower() != ".png" or any(":" in part for part in path.parts)):
            raise ValueError(f"prediction path must be a relative PNG under predictions/: {raw}")
        resolved = self.output_dir.joinpath(*path.parts).resolve()
        if not resolved.is_relative_to(self.output_dir / "predictions"):
            raise ValueError(f"prediction path escapes output directory: {raw}")
        return resolved

    @staticmethod
    def _encode(mask: np.ndarray, path: Path) -> None:
        Image.fromarray(mask).save(path, format='PNG')

    def write(self, predictions: Iterable[Prediction], pool=None, queue_depth: int = 64) -> Path:
        if type(queue_depth) is not int or queue_depth <= 0:
            raise ValueError('queue_depth must be a positive integer')
        for directory in {path.parent for path in self.paths.values()}:
            directory.mkdir(parents=True, exist_ok=True)
        pending = deque()
        for prediction in predictions:
            if prediction.image_path not in self.paths or prediction.image_path in self._written:
                raise ValueError(f"unexpected or duplicate prediction: {prediction.image_path}")
            mask = prediction.mask
            if mask.ndim != 2 or mask.dtype != np.uint8 or not np.isin(mask, (0, 255)).all():
                raise ValueError("prediction must be a single-channel uint8 mask containing only 0/255")
            path = self.paths[prediction.image_path]
            if pool is None:
                self._encode(mask, path)
            else:
                pending.append(pool.submit(self._encode, mask, path))
                if len(pending) >= queue_depth:
                    pending.popleft().result()
            self._written.add(prediction.image_path)
        for future in pending:
            future.result()
        if self._written != set(self.paths):
            raise ValueError("predictions are missing for some template rows")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.output_dir / "submission.csv"
        ConsoleProgress.info(f"Submission: сохранение таблицы {csv_path}")
        self.template.to_csv(csv_path, index=False)
        return csv_path


@dataclass(frozen=True)
class InferenceConfig:
    model: ModelConfig
    image_size: int
    seed: int
    data_path: Path
    device: str
    amp: str
    batch_size: int
    workers: int
    aux_weight: float = .4

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> "InferenceConfig":
        """Normalize current and explicitly supported historical run snapshots."""
        config = ExperimentConfig.from_dict(SnapshotAdapter.normalize(snapshot))
        return cls(config.model, config.dataset.image_size, config.seed, config.paths.data_path,
                   config.train.device, config.train.amp, config.train.batch_size, config.train.workers,
                   config.loss.aux_weight)



def create_submission(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    thresholds: ThresholdConfig | None = None,
    data_path: str | Path | None = None,
    template_path: str | Path | None = None,
    device: str | None = None,
    checkpoint_name: str = 'best.pt',
    batch_size: int | None = None,
    workers: int | None = None,
    post_workers: int = 8,
) -> Path:
    """Load the selected checkpoint (EMA preferred) and write the submission."""
    if checkpoint_name not in {'best.pt', 'last.pt'}:
        raise ValueError('checkpoint_name must be best.pt or last.pt')
    ConsoleProgress.info(f"Submission: загрузка конфигурации запуска {run_dir}")
    run = Run.open(run_dir)
    config = InferenceConfig.from_snapshot(run.snapshot)
    if thresholds is None:
        RetunedSelection.verify(run.summary, run.snapshot, run.dir / 'ckpt' / checkpoint_name)
        thresholds = ThresholdConfig.from_summary(run.summary, run.snapshot)
    batch_size = config.batch_size if batch_size is None else batch_size
    workers = config.workers if workers is None else workers
    if (type(batch_size) is not int or batch_size <= 0 or type(workers) is not int or workers < 0
            or type(post_workers) is not int or post_workers < 0):
        raise ValueError('batch_size must be positive; workers and post_workers non-negative integers')
    ConsoleProgress.info(
        f"Submission: пороги mask={thresholds.mask_threshold:g}, "
        f"cls={thresholds.cls_threshold:g}, min_area={thresholds.min_area:g}, area_cap={thresholds.area_cap:g}"
    )
    ConsoleProgress.info("Submission: чтение тестовых данных и шаблона")
    workspace = DataWorkspace(Path(data_path) if data_path is not None else config.data_path)
    test_rows = workspace.test_csv
    template = pd.read_csv(template_path or workspace.test_root / "submission.csv")
    writer = SubmissionWriter(template, test_rows, output_dir)
    dataset = AIIJCDataset(workspace, test_rows, False, config.image_size, config.seed, mode='test')
    inference_device = torch.device(device or config.device)
    ConsoleProgress.info(
        f"Submission: изображений {len(test_rows)}, устройство {inference_device}, "
        f"batch_size={batch_size}, workers={workers}, post_workers={post_workers}"
    )
    amp = AmpContext(inference_device, torch.bfloat16 if config.amp == "bf16" else torch.float16,
                     inference_device.type == "cuda" and config.amp != "off", False)
    checkpoint_path = run.dir / "ckpt" / checkpoint_name
    ConsoleProgress.info(f"Submission: загрузка checkpoint {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    weights = "ema" if checkpoint.get("ema") is not None else "model"
    ConsoleProgress.info(f"Submission: создание модели и применение весов {weights}")
    model = build_model(config.model, aux_weight=config.aux_weight, pretrained=False)
    model.load_state_dict(checkpoint[weights])
    del checkpoint
    if inference_device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        collate_fn=ValidationCollator(), pin_memory=inference_device.type == 'cuda',
                        persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None)
    ConsoleProgress.info(f"Submission: перенос модели на {inference_device}")
    predictor = Predictor(model, thresholds, amp)
    batches = ConsoleProgress.iterate(loader, "Submission: предсказание и сохранение PNG (батчи)")
    if post_workers:
        with ThreadPoolExecutor(max_workers=post_workers, thread_name_prefix='postprocess') as pool:
            csv_path = writer.write(predictor.predict(batches, pool=pool), pool=pool)
    else:
        csv_path = writer.write(predictor.predict(batches))
    ConsoleProgress.info(f"Submission: готово, масок {len(test_rows)}, результат {csv_path.parent}")
    return csv_path
