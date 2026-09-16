"""Папка прогона: ручка, которая пишет и читает документированный формат.

Формат тот же, что был у `train.run`, и меняться не будет — на нём стоят все
накопленные прогоны:

    <прогон>/
      config.yaml          непрозрачный снапшот: что вы захотели запомнить
      metrics.jsonl        источник правды, дописывается построчно
      metrics.csv          то же для Excel, перезаписывается на close()
      train.log            текстовый журнал
      summary.json         итоги, дописываются в несколько заходов
      ckpt/                чекпоинты
      oof/val.npz          гистограммы валидации
      oof/val_rows.parquet строки валидации в том же порядке
      tb/                  TensorBoard
      preds/               картинки для глаз

Библиотека не заглядывает в `config.yaml`: что там лежит — дело того, кто писал.

Писатели — глаголы (`log`, `save_*`), читатели — свойства (`history`,
`snapshot`, `summary`). Иначе `snapshot` пришлось бы делать и методом, и
свойством одновременно.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .metric import (
    DEFAULT_AREA_GRID,
    DEFAULT_CAP_GRID,
    DEFAULT_CLS_GRID,
    DEFAULT_MASK_GRID,
    AICAccumulator,
    AICResult,
)
from .notes import NOTES_NAME, notes_template
from .validation import ValidationResult

#: подпапки, которые создаются сразу: пути на них ссылаются до первой записи
SUBDIRS = ("ckpt", "oof", "tb", "preds")


@dataclass(frozen=True)
class Eval:
    """Оценка на выборке: аккумулятор и имена кадров в том же порядке.

    Пара, а не два аргумента: величины берутся из аккумулятора по позиции, а
    совпадение кадров ищется по stem'ам. Разъехались длины — сравнение молча
    смешает разные кадры, и разница будет мерить не гипотезу, а рассинхрон.
    Проверка здесь делает такое состояние непредставимым.
    """

    acc: AICAccumulator
    stems: np.ndarray

    def __post_init__(self) -> None:
        stems = np.asarray(self.stems)
        object.__setattr__(self, "stems", stems)
        if stems.size != len(self.acc):
            raise ValueError(
                f"в аккумуляторе {len(self.acc)} кадров, а stem'ов {stems.size}"
            )
        if len(np.unique(stems)) != stems.size:
            raise ValueError("stem'ы повторяются — по ним нельзя сопоставить кадры")

    def __len__(self) -> int:
        return len(self.acc)

    def best(
        self,
        mask_thresholds=None,
        cls_thresholds=DEFAULT_CLS_GRID,
        min_areas=DEFAULT_AREA_GRID,
        area_caps=DEFAULT_CAP_GRID,
    ) -> AICResult:
        grid = list(DEFAULT_MASK_GRID) if mask_thresholds is None else list(mask_thresholds)
        return self.acc.best(grid, list(cls_thresholds), list(min_areas), list(area_caps))


class Run:
    """Ручка на папку прогона."""

    def __init__(self, run_dir: str | Path, *, tensorboard: bool = False) -> None:
        self.dir = Path(run_dir)
        self.jsonl_path = self.dir / "metrics.jsonl"
        self.csv_path = self.dir / "metrics.csv"
        self.log_path = self.dir / "train.log"
        self.summary_path = self.dir / "summary.json"
        self.notes_path = self.dir / NOTES_NAME
        self.start = time.time()
        self.writer = None

        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(self.dir / "tb"))
            except Exception as exc:  # tensorboard не обязателен
                self.info(f"TensorBoard отключён: {exc}")

    # --- создание и открытие ---------------------------------------------

    @classmethod
    def create(
        cls,
        runs_root: str | Path,
        name: str,
        *,
        resume: bool = False,
        tensorboard: bool = True,
    ) -> Run:
        """`<runs_root>/<name>`; занятое имя без `resume` получает суффикс времени.

        Затирать чужую папку нельзя ни при каких обстоятельствах: в ней лежат
        часы обучения, и восстановить их неоткуда.
        """
        runs_root = Path(runs_root)
        runs_root.mkdir(parents=True, exist_ok=True)
        run_dir = runs_root / name
        if run_dir.exists() and not resume:
            run_dir = runs_root / f"{name}__{time.strftime('%m%d-%H%M%S')}"
        for sub in SUBDIRS:
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        run = cls(run_dir, tensorboard=tensorboard)
        run.ensure_notes()
        return run

    def ensure_notes(self) -> Path:
        """Заготовка карточки. Написанную карточку не трогает никогда.

        Та же причина, по которой `create` не затирает чужую папку: в карточке
        лежит вывод эксперимента, и восстановить его неоткуда.
        """
        if not self.notes_path.exists():
            self.notes_path.write_text(notes_template(self.dir.name), encoding="utf-8")
        return self.notes_path

    @classmethod
    def open(cls, run_dir: str | Path) -> Run:
        """Чужая папка на чтение. TensorBoard не поднимается."""
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"нет папки прогона {run_dir}")
        return cls(run_dir, tensorboard=False)

    def __repr__(self) -> str:
        return f"Run({str(self.dir)!r})"

    # --- запись ------------------------------------------------------------

    def info(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def log(self, step: int, metrics: Mapping[str, Any], prefix: str = "") -> None:
        """Строка в `metrics.jsonl` (+ TensorBoard). `step` и `elapsed_s` свои.

        Ось показов библиотека не считает: если сравнение прогонов пойдёт по
        числу показов, а не по номеру эпохи, логируйте `samples` обычной
        метрикой — см. `Run.curve`.
        """
        record: dict[str, Any] = {
            "step": int(step),
            "elapsed_s": round(time.time() - self.start, 1),
        }
        for key, value in metrics.items():
            record[f"{prefix}{key}" if prefix else key] = value

        self.dir.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        if self.writer is not None:
            for key, value in record.items():
                if isinstance(value, (int, float)) and key not in {"step", "elapsed_s"}:
                    self.writer.add_scalar(key.replace("/", "_").replace("_", "/", 1), value, step)
            self.writer.flush()

    def save_snapshot(self, obj: Mapping[str, Any], name: str = "config.yaml") -> Path:
        """Записать снапшот как есть. Библиотека внутрь не смотрит."""
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(dict(obj), fh, allow_unicode=True, sort_keys=False)
        return path

    def save_summary(self, patch: Mapping[str, Any]) -> Path:
        """Дописать в `summary.json`, слив по ВЕРХНЕМУ уровню ключей.

        Мерж, а не перезапись: сводку заполняют в несколько заходов — обучение,
        потом калибровка, потом бюджет. Вложенные словари заменяются целиком:
        сливать `best` по частям некому и незачем, а частичный `best` был бы
        опаснее отсутствующего.
        """
        current = self.summary
        current.update(dict(patch))
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(
            json.dumps(current, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return self.summary_path

    def flush_csv(self) -> None:
        if not self.jsonl_path.exists():
            return
        self._history_frame().to_csv(self.csv_path, index=False)

    def close(self) -> None:
        self.flush_csv()
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def __enter__(self) -> Run:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # --- чтение -------------------------------------------------------------

    def _history_frame(self) -> pd.DataFrame:
        if not self.jsonl_path.exists():
            return pd.DataFrame()
        rows = [
            json.loads(line)
            for line in self.jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return pd.DataFrame(rows)

    @property
    def summary(self) -> dict:
        """`summary.json` или пустой словарь."""
        if not self.summary_path.exists():
            return {}
        return json.loads(self.summary_path.read_text(encoding="utf-8")) or {}

    @property
    def history(self) -> pd.DataFrame:
        """`metrics.jsonl` таблицей. Пустой DataFrame, если лога ещё нет."""
        return self._history_frame()

    @property
    def snapshot(self) -> dict:
        """`config.yaml` как есть. Библиотека внутрь не смотрит."""
        path = self.dir / "config.yaml"
        if not path.exists():
            return {}
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def curve(
        self,
        y: str,
        x: str | tuple[str, float] = "samples",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Две оси из лога — для `stats.gate_check` и для графиков.

        `x` — имя колонки либо пара `(колонка, множитель)`. Пара нужна старым
        прогонам: числа показов они не логировали, и ось строится из номера
        эпохи как `(step + 1) * samples_per_epoch`. Множитель передаёт вызывающий,
        потому что только он знает, где у него лежит размер эпохи.

        Отсутствие колонки — ошибка, а не пустая кривая. Молчаливый ноль тут
        особенно дорог: `gate_check` на нулевой оси отвечает «вне кривой
        эталона» на каждой эпохе, то есть гейт выключается на весь прогон, и
        узнать об этом неоткуда.
        """
        frame = self._history_frame()
        if frame.empty:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

        if y not in frame.columns:
            raise KeyError(
                f"в логе {self.jsonl_path} нет метрики {y!r}; "
                f"есть: {', '.join(sorted(frame.columns))}"
            )

        # сдвиг на единицу только в форме с множителем: эпоха номер 0 — это уже
        # один пройденный размер эпохи, а не ноль показов. Ровно так строил ось
        # прежний load_reference, и совместимость чисел держится на этом
        column, scale, shift = (x, 1.0, 0.0) if isinstance(x, str) else (x[0], float(x[1]), 1.0)
        if column not in frame.columns:
            raise KeyError(
                f"в логе {self.jsonl_path} нет колонки оси {column!r}. "
                f"Логируйте её обычной метрикой (run.log(epoch, {{'samples': ...}})) "
                f'или постройте ось из номера эпохи: x=("step", размер_эпохи)'
            )

        rows = frame[frame[y].notna() & frame[column].notna()]
        xs = (rows[column].to_numpy(dtype=float) + shift) * scale
        return xs, rows[y].to_numpy(dtype=float)

    # --- оценка -------------------------------------------------------------

    def save_eval(
        self, result: ValidationResult | AICAccumulator, rows: pd.DataFrame, name: str = "val"
    ) -> Path:
        """`oof/<name>.npz` + `oof/<name>_rows.parquet`.

        Строки пишутся целиком, не только stem: разрезы по домену, генератору и
        площади GT в `analysis` берутся отсюда.
        """
        acc = result.accumulator if isinstance(result, ValidationResult) else result
        if "stem" not in rows.columns:
            raise ValueError(f"в таблице строк нет колонки stem (есть {list(rows.columns)})")
        if len(rows) != len(acc):
            raise ValueError(
                f"в аккумуляторе {len(acc)} кадров, а строк {len(rows)}. Сопоставление "
                "идёт по позиции, и на разной длине оно смешало бы разные кадры"
            )
        oof = self.dir / "oof"
        oof.mkdir(parents=True, exist_ok=True)
        acc.save(oof / f"{name}.npz")
        rows.to_parquet(oof / f"{name}_rows.parquet", index=False)
        if isinstance(result, ValidationResult):
            summary = {f"{name}_best": result.tuned.as_dict()}
            if name == "val":
                summary.update(best=result.tuned.as_dict(), best_aic=result.tuned.aic)
                summary["validation_resolution"] = result.resolution
            self.save_summary(summary)
        return oof / f"{name}.npz"

    def load_rows(self, name: str = "val") -> pd.DataFrame:
        path = self.dir / "oof" / f"{name}_rows.parquet"
        if not path.exists():
            raise FileNotFoundError(f"нет {path}")
        return pd.read_parquet(path)

    def load_eval(self, name: str = "val") -> Eval:
        path = self.dir / "oof" / f"{name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"нет {path}")
        acc = AICAccumulator.load(path)
        rows = self.load_rows(name)
        try:
            return Eval(acc, rows["stem"].to_numpy())
        except ValueError as error:
            # к общей формулировке добавляем, ЧТО именно разошлось на диске:
            # без имён файлов сообщение не подсказывает, что чинить
            raise ValueError(
                f"{self.dir.name}: в {name}.npz {len(acc)} кадров, а в "
                f"{name}_rows.parquet {len(rows)} строк. Сопоставление идёт по "
                f"stem'ам в порядке этих строк, и на разной длине оно молча "
                f"смешало бы разные кадры"
            ) from error

    def operating_point(self, name: str = "val") -> tuple[float, float, float, float]:
        """`(mask_threshold, cls_threshold, min_area, area_cap)` из сводки, иначе свипом.

        Остаток прежнего `load_reference`: остальная его работа разошлась по
        `Run.open`, `history` и `load_eval`.
        """
        best = self.summary.get(f"{name}_best") or (
            self.summary.get("best") if name == "val" else None
        ) or {}
        if {"mask_threshold", "cls_threshold", "min_area"} <= set(best):
            return (
                float(best["mask_threshold"]),
                float(best["cls_threshold"]),
                float(best["min_area"]),
                float(best.get("area_cap", 0.0)),
            )
        found = self.load_eval(name).best()
        return (found.mask_threshold, found.cls_threshold, found.min_area, found.area_cap)

    # --- чекпоинты -----------------------------------------------------------

    def save_state(self, state: Mapping[str, Any], name: str = "last.pt") -> Path:
        """Записать состояние атомарно. Что в нём лежит — решает вызывающий.

        Библиотека не знает ни про EMA, ни про scaler, ни про шедулер. Запись
        идёт во временный файл рядом и `os.replace`: прогон, убитый посреди
        записи, не должен оставлять чекпоинт, с которого потом не поднимется
        resume.
        """
        import torch

        ckpt = self.dir / "ckpt"
        ckpt.mkdir(parents=True, exist_ok=True)
        target = ckpt / name
        tmp = target.with_suffix(target.suffix + ".tmp")
        torch.save(dict(state), tmp)
        os.replace(tmp, target)
        return target

    def load_state(self, name: str = "last.pt", map_location: Any = "cpu") -> dict:
        import torch

        path = self.dir / "ckpt" / name
        if not path.exists():
            raise FileNotFoundError(f"нет чекпоинта {path}")
        return torch.load(str(path), map_location=map_location, weights_only=False)
