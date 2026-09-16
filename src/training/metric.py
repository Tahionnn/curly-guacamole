"""Метрика AIC Score — ровно по условию задачи.

    Dice_pos = mean_{i in positive} 2|P_i ∩ G_i| / (|P_i| + |G_i| + 1e-6)
    FPR_neg  = mean_{j in negative} 1[ |P_j| / (H_j * W_j) >= 0.01 ]
    AIC      = 2 * Dice_pos * (1 - FPR_neg) / (Dice_pos + (1 - FPR_neg))

Позитив/негатив определяется по GT: негатив — кадр, где в GT нет изменений.

Главный рабочий инструмент здесь — `AICAccumulator`. Он на лету сжимает
предсказанные вероятности в гистограммы, поэтому по одному проходу валидации
можно потом за доли секунды перебрать любую сетку порогов бинаризации,
порогов классификатора и правил по минимальной площади — не гоняя модель заново.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

EPS = 1e-6
FP_AREA_THRESHOLD = 0.01  # площадь >= 1% кадра => ложная тревога


def threshold_bin(threshold: float, n_bins: int) -> int:
    """Floor to the histogram grid, preserving boundaries such as 29 / 100."""
    scaled = float(threshold) * n_bins
    nearest = round(scaled)
    if math.isclose(scaled, nearest, rel_tol=0., abs_tol=1e-10):
        scaled = nearest
    return min(max(int(scaled), 0), n_bins - 1)


def harmonic_aic(dice_pos: float, fpr_neg: float) -> float:
    """Итоговая свёртка двух компонент. 0, если любая из них нулевая."""
    a = float(dice_pos)
    b = 1.0 - float(fpr_neg)
    if a + b <= 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)


def dice_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice для двух бинарных масок одного размера."""
    pred_sum = float(np.count_nonzero(pred))
    gt_sum = float(np.count_nonzero(gt))
    inter = float(np.count_nonzero(np.logical_and(pred, gt)))
    return 2.0 * inter / (pred_sum + gt_sum + EPS)


@dataclass
class AICResult:
    aic: float
    dice_pos: float
    fpr_neg: float
    n_pos: int
    n_neg: int
    mask_threshold: float = 0.5
    cls_threshold: float = 0.0
    min_area: float = 0.0
    small_mask_weight: float = 1.0
    area_cap: float = 0.0

    def as_dict(self) -> dict:
        return {
            "aic": self.aic,
            "dice_pos": self.dice_pos,
            "fpr_neg": self.fpr_neg,
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
            "mask_threshold": self.mask_threshold,
            "cls_threshold": self.cls_threshold,
            "min_area": self.min_area,
            "area_cap": self.area_cap,
            "small_mask_weight": self.small_mask_weight,
        }

    def __str__(self) -> str:
        label = 'AIC' if self.small_mask_weight == 1.0 else f'weighted AIC (small={self.small_mask_weight:g})'
        cap = f" cap={self.area_cap:.4f}" if self.area_cap > 0 else ""
        return (
            f"{label}={self.aic:.4f} (Dice_pos={self.dice_pos:.4f}, FPR_neg={self.fpr_neg:.4f}) "
            f"@ thr={self.mask_threshold:.3f} cls={self.cls_threshold:.3f} "
            f"min_area={self.min_area:.3f}{cap} | pos={self.n_pos} neg={self.n_neg}"
        )


def score_masks(
    preds: Iterable[np.ndarray],
    gts: Iterable[np.ndarray],
    *,
    pred_positive_value: int = 128,
    gt_positive_value: int = 128,
) -> AICResult:
    """Прямой честный подсчёт по готовым маскам (для тестов и финальной сверки).

    Маски принимаются как uint8 0/255 либо как bool. Пороги сравнения `>=`.
    """
    dices: list[float] = []
    false_alarms: list[float] = []

    for pred, gt in zip(preds, gts, strict=True):
        pred_bin = pred >= pred_positive_value if pred.dtype != bool else pred
        gt_bin = gt >= gt_positive_value if gt.dtype != bool else gt
        if pred_bin.shape != gt_bin.shape:
            raise ValueError(f"размеры не совпадают: pred {pred_bin.shape} vs gt {gt_bin.shape}")

        if np.count_nonzero(gt_bin) > 0:
            dices.append(dice_binary(pred_bin, gt_bin))
        else:
            area = np.count_nonzero(pred_bin) / float(gt_bin.size)
            false_alarms.append(float(area >= FP_AREA_THRESHOLD))

    dice_pos = float(np.mean(dices)) if dices else 0.0
    fpr_neg = float(np.mean(false_alarms)) if false_alarms else 0.0
    return AICResult(
        aic=harmonic_aic(dice_pos, fpr_neg),
        dice_pos=dice_pos,
        fpr_neg=fpr_neg,
        n_pos=len(dices),
        n_neg=len(false_alarms),
    )


def cap_bins(pred_counts: np.ndarray, n_pixels: np.ndarray, area_cap: float) -> np.ndarray:
    """Нижний бин, на котором площадь предсказания уже меньше `area_cap`.

    Строка — кадр. Если даже верхний бин не укладывается в лимит (скажем, весь
    кадр предсказан с вероятностью 1), возвращается последний бин, а решение
    «обнулить» принимает вызывающий код.
    """
    area = pred_counts / np.maximum(n_pixels, 1.0)[:, None]
    under = area < area_cap
    return np.where(under.any(axis=1), under.argmax(axis=1), pred_counts.shape[1] - 1)


def operating_bins(
    pred_counts: np.ndarray,
    n_pixels: np.ndarray,
    cls_prob: np.ndarray,
    bin_index: int,
    cls_threshold: float,
    min_area: float,
    area_cap: float = 0.0,
    cap_index: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Бин порога на каждый кадр и флаг «обнулить маску» для одной рабочей точки.

    Единственная реализация правила постобработки: и свип порогов, и по-кадровый
    отчёт в `eval.diagnostics` читают её. Пока прочтений рабочей точки было два,
    они однажды разошлись бы — и отбор шёл бы не по той метрике, по которой
    считается вердикт.

    `area_cap > 0` меняет действие cls-гейта: вместо обнуления кадра порог
    поднимается ровно настолько, чтобы площадь ушла под `area_cap`. Для FPR это
    то же самое при cap <= 0.01, но у позитивов, ошибочно
    попавших под гейт, остаётся ненулевой Dice вместо нуля.

    `cap_index` — уже посчитанный `cap_bins` для этого `area_cap`; он не зависит
    от перебираемой точки, и свип считает его один раз на сетку.
    """
    rows = np.arange(pred_counts.shape[0])
    bins = np.full(pred_counts.shape[0], int(bin_index), dtype=np.int64)
    gated = cls_prob < cls_threshold
    # min_area читается по общему порогу, до поджатия: правило про «слишком
    # мелкое предсказание» не должно срабатывать от самого поджатия.
    blank = pred_counts[rows, bins] / np.maximum(n_pixels, 1.0) < min_area
    if area_cap <= 0.0:
        return bins, blank | gated
    if cap_index is None:
        cap_index = cap_bins(pred_counts, n_pixels, area_cap)
    bins = np.where(gated, np.maximum(bins, cap_index), bins)
    capped = pred_counts[rows, bins] / np.maximum(n_pixels, 1.0)
    return bins, blank | (gated & (capped >= area_cap))


@dataclass
class AICAccumulator:
    """Копит по-картиночную статистику в сжатом виде для последующего свипа порогов.

    На каждый кадр хранится две гистограммы предсказанных вероятностей
    (по всем пикселям и по пикселям GT) размера `n_bins`. Из них восстанавливается
    |P_t| и |P_t ∩ G| для любого порога t из сетки k / n_bins.
    small_mask_weight меняет вес Dice позитивов с GT-площадью (1%, 5%].
    По умолчанию вес 1 сохраняет официальную метрику; FPR не взвешивается.
    """

    n_bins: int = 256
    hist_all: list[np.ndarray] = field(default_factory=list)
    hist_gt: list[np.ndarray] = field(default_factory=list)
    gt_sum: list[int] = field(default_factory=list)
    n_pixels: list[int] = field(default_factory=list)
    cls_prob: list[float] = field(default_factory=list)
    small_mask_weight: float = 1.0

    def __post_init__(self):
        if isinstance(self.small_mask_weight, bool) or not np.isfinite(self.small_mask_weight) or self.small_mask_weight <= 0:
            raise ValueError('small_mask_weight must be finite and positive')

    @property
    def thresholds(self) -> np.ndarray:
        return np.arange(self.n_bins, dtype=np.float64) / self.n_bins

    def update(
        self,
        probs: np.ndarray | object,
        gts: np.ndarray | object,
        cls_probs: np.ndarray | object | None = None,
    ) -> None:
        """Добавить батч. `probs`/`gts` — (B, H, W) или (B, 1, H, W), numpy или torch.

        `probs` — вероятности в [0, 1], `gts` — бинарные (0/1).
        """
        probs_np = _to_numpy(probs)
        gts_np = _to_numpy(gts)
        probs_np = probs_np.reshape(probs_np.shape[0], -1)
        gts_np = gts_np.reshape(gts_np.shape[0], -1)

        if cls_probs is None:
            cls_np = np.ones(probs_np.shape[0], dtype=np.float32)
        else:
            cls_np = _to_numpy(cls_probs).reshape(-1).astype(np.float32)

        idx_all = np.clip((probs_np * self.n_bins).astype(np.int32), 0, self.n_bins - 1)
        gt_bool = gts_np > 0.5

        for i in range(probs_np.shape[0]):
            row = idx_all[i]
            self.hist_all.append(np.bincount(row, minlength=self.n_bins).astype(np.int64))
            self.hist_gt.append(
                np.bincount(row[gt_bool[i]], minlength=self.n_bins).astype(np.int64)
            )
            self.gt_sum.append(int(gt_bool[i].sum()))
            self.n_pixels.append(int(row.size))
            self.cls_prob.append(float(cls_np[i]))

    def update_hist(
        self,
        hist_all: np.ndarray,
        hist_gt: np.ndarray,
        gt_sum: np.ndarray,
        n_pixels: np.ndarray,
        cls_prob: np.ndarray,
    ) -> None:
        """Приём уже готовых гистограмм (их быстрее посчитать на GPU в engine)."""
        for i in range(hist_all.shape[0]):
            self.hist_all.append(hist_all[i].astype(np.int64))
            self.hist_gt.append(hist_gt[i].astype(np.int64))
            self.gt_sum.append(int(gt_sum[i]))
            self.n_pixels.append(int(n_pixels[i]))
            self.cls_prob.append(float(cls_prob[i]))

    def __len__(self) -> int:
        return len(self.gt_sum)

    # --- расчёт ---------------------------------------------------------

    def tables(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(pred_counts, inter_counts, gt_sum, n_pixels, cls_prob), формы (N, n_bins) и (N,).

        Публичный, а не приватный: разбор валидации по корзинам площади
        (`analysis.OofView`) строится ровно на этих таблицах. Пока метод был
        приватным, у разворота гистограмм было две реализации, и однажды они
        разошлись бы — а вердикты тогда считались бы не по той метрике, по
        которой отбираются модели.
        """
        if not self.gt_sum:
            raise ValueError("аккумулятор пуст — нечего считать")
        hist_all = np.stack(self.hist_all)  # (N, n_bins)
        hist_gt = np.stack(self.hist_gt)
        # |P_t| для t = k/n_bins  <=>  сумма бинов с индексом >= k
        pred_counts = np.cumsum(hist_all[:, ::-1], axis=1)[:, ::-1]
        inter_counts = np.cumsum(hist_gt[:, ::-1], axis=1)[:, ::-1]
        return (
            pred_counts,
            inter_counts,
            np.asarray(self.gt_sum, dtype=np.float64),
            np.asarray(self.n_pixels, dtype=np.float64),
            np.asarray(self.cls_prob, dtype=np.float64),
        )

    def evaluate(
        self,
        mask_threshold: float = 0.5,
        cls_threshold: float = 0.0,
        min_area: float = 0.0,
        area_cap: float = 0.0,
    ) -> AICResult:
        grid = self.sweep(
            mask_thresholds=[mask_threshold],
            cls_thresholds=[cls_threshold],
            min_areas=[min_area],
            area_caps=[area_cap],
        )
        return grid[0]

    def sweep(
        self,
        mask_thresholds: Sequence[float] | None = None,
        cls_thresholds: Sequence[float] = (0.0,),
        min_areas: Sequence[float] = (0.0,),
        area_caps: Sequence[float] = (0.0,),
    ) -> list[AICResult]:
        """Перебор сетки постобработки. Возвращает список, отсортированный по AIC убыв.

        `cls_threshold` — порог aux-головы: если вероятность «кадр изменён» ниже,
        маска обнуляется целиком.
        `min_area` — доля кадра: предсказания меньшей площади обнуляются. Заметьте,
        что при `min_area <= 0.01` FPR не меняется вовсе: ложная тревога считается
        от 1% кадра, поэтому обнуляются только кадры, которые и так не были
        ложной тревогой, а Dice позитивов при этом падает.
        `area_cap` — что делает cls-гейт: при 0 обнуляет кадр, иначе поднимает его
        порог ровно настолько, чтобы площадь ушла под `area_cap`. Правило считает
        `operating_bins`, ту же функцию читает inference.
        """
        pred_counts, inter_counts, gt_sum, n_pixels, cls_prob = self.tables()
        is_pos = gt_sum > 0
        # Weight images by original GT area, never by their predicted mask area.
        gt_area = gt_sum[is_pos] / n_pixels[is_pos]
        weights = np.where((gt_area > .01) & (gt_area <= .05), self.small_mask_weight, 1.0)

        if mask_thresholds is None:
            mask_thresholds = self.thresholds
        bin_idx = [threshold_bin(threshold, self.n_bins) for threshold in mask_thresholds]

        rows = np.arange(pred_counts.shape[0])
        # cap_bins depends only on area_cap, never on the point being swept.
        cap_index = {float(cap): cap_bins(pred_counts, n_pixels, float(cap))
                     for cap in area_caps if float(cap) > 0.0}

        results: list[AICResult] = []
        for k in bin_idx:
            for cls_thr in cls_thresholds:
                for min_area in min_areas:
                    for area_cap in area_caps:
                        bins, blank = operating_bins(
                            pred_counts, n_pixels, cls_prob, k, cls_thr, min_area,
                            area_cap, cap_index.get(float(area_cap)),
                        )
                        keep = ~blank
                        pred_k = np.where(keep, pred_counts[rows, bins].astype(np.float64), 0.0)
                        inter_k = np.where(keep, inter_counts[rows, bins].astype(np.float64), 0.0)
                        area_k = pred_k / np.maximum(n_pixels, 1.0)

                        dice = 2.0 * inter_k / (pred_k + gt_sum + EPS)
                        dice_pos = float(np.average(dice[is_pos], weights=weights)) if is_pos.any() else 0.0
                        if (~is_pos).any():
                            fpr_neg = float((area_k[~is_pos] >= FP_AREA_THRESHOLD).mean())
                        else:
                            fpr_neg = 0.0

                        results.append(
                            AICResult(
                                aic=harmonic_aic(dice_pos, fpr_neg),
                                dice_pos=dice_pos,
                                fpr_neg=fpr_neg,
                                n_pos=int(is_pos.sum()),
                                n_neg=int((~is_pos).sum()),
                                # Histograms only resolve boundaries k / n_bins.
                                # Persist the boundary actually used for inference parity.
                                mask_threshold=float(k / self.n_bins),
                                cls_threshold=float(cls_thr),
                                min_area=float(min_area),
                                area_cap=float(area_cap),
                                small_mask_weight=self.small_mask_weight,
                            )
                        )

        results.sort(key=lambda r: r.aic, reverse=True)
        return results

    def best(
        self,
        mask_thresholds: Sequence[float] | None = None,
        cls_thresholds: Sequence[float] = (0.0,),
        min_areas: Sequence[float] = (0.0,),
        area_caps: Sequence[float] = (0.0,),
    ) -> AICResult:
        return self.sweep(mask_thresholds, cls_thresholds, min_areas, area_caps)[0]

    # --- сохранение / загрузка -----------------------------------------

    def save(self, path) -> None:
        np.savez_compressed(
            str(path),
            n_bins=self.n_bins,
            small_mask_weight=self.small_mask_weight,
            hist_all=np.stack(self.hist_all).astype(np.int32),
            hist_gt=np.stack(self.hist_gt).astype(np.int32),
            gt_sum=np.asarray(self.gt_sum, dtype=np.int64),
            n_pixels=np.asarray(self.n_pixels, dtype=np.int64),
            cls_prob=np.asarray(self.cls_prob, dtype=np.float32),
        )

    @classmethod
    def load(cls, path) -> AICAccumulator:
        data = np.load(str(path))
        acc = cls(n_bins=int(data["n_bins"]), small_mask_weight=float(data.get("small_mask_weight", 1.0)))
        acc.hist_all = list(data["hist_all"].astype(np.int64))
        acc.hist_gt = list(data["hist_gt"].astype(np.int64))
        acc.gt_sum = data["gt_sum"].tolist()
        acc.n_pixels = data["n_pixels"].tolist()
        acc.cls_prob = data["cls_prob"].tolist()
        return acc


def _to_numpy(array) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    detach = getattr(array, "detach", None)
    if detach is not None:  # torch.Tensor
        return detach().float().cpu().numpy()
    return np.asarray(array)


DEFAULT_MASK_GRID = tuple(np.round(np.arange(0.05, 0.96, 0.025), 4).tolist())
DEFAULT_CLS_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
DEFAULT_AREA_GRID = (0.0, 0.005, 0.01, 0.02, 0.03)
# Compare several caps; Dice need not improve monotonically as retained area grows.
DEFAULT_CAP_GRID = (0.0, 0.008, 0.0095, 0.01)
