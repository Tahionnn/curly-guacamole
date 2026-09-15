# BiFPN перед EMCAD

Путь: B2-Li → JPEG fusion → DG patch/edge → BiFPN → EMCAD → маска.
Классификационная голова читает прежний выход DG до BiFPN. Cross-attention и
обратный attention 32→4 отключены в исходных конфигурациях этой серии.
Вариант `w64_r2_cross32` добавляет attention 16→32 после JPEG/DG fusion,
до BiFPN; обратный attention 32→4 остаётся отключён.

BiFPN получает четыре карты strides 4/8/16/32, проецирует их в общую ширину,
выполняет проход 32→16→8→4, затем 4→8→16→32. Промежуточные узлы обратного
прохода смешивают исходную карту, результат первого прохода и нижний уровень.
Веса связей — обучаемые скаляры с ReLU и нормировкой на сумму + 1e-4;
они не зависят от изображения. Каждый узел: SiLU → depthwise 3×3 → pointwise
1×1 → BN. Повторения имеют независимые веса. Upsample nearest, downsample
max-pool 3×3/stride 2; поддержаны нечётные и прямоугольные сетки.

Выходные проекции возвращают каналы 64/128/320/512 для неизменённого EMCAD.
Внешнего нулевого gate нет: включённый BiFPN сразу меняет признаки декодера.
По умолчанию bifpn_repeats=0, старые модели и их state-dict keys сохраняются.
Новый модуль создаётся после существующих, сохраняя их seeded initialization.

Это адаптация BiFPN как промежуточного модуля, не воспроизведение EfficientDet:
четыре уровня вместо detection-пирамиды, общие входные проекции и обратные
проекции для EMCAD. Loss остаётся прежним: mask/aux/classification и DG
patch/edge. DG-головы расположены до BiFPN; он получает градиенты mask/aux.

Источник: https://arxiv.org/abs/1911.09070 (раздел 3).

## Измерения

Полный eval forward через FlopCounterMode, native JPEG 1080×1920, включая
обе проекции каналов, JPEG-ветку, DG и EMCAD. Meta-замеры при RGB 704
подтверждены на CUDA/RTX 3070. Контроль без BiFPN: 84.625 GFLOPs при RGB 704.

| Ширина | Повторов | Параметры BiFPN | GFLOPs при 704 | Добавка | Max RGB (шаг 8) | GFLOPs на max | Следующий размер / GFLOPs |
|---|---|---|---|---|---|---|---|
| 64 | 1 | 162446 | 86.083 | 1.458 | 752 | 98.160 | 760 / 100.621 |
| 64 | 2 | 191260 | 86.558 | 1.933 | 752 | 98.702 | 760 / 101.175 |
| 128 | 1 | 371982 | 88.374 | 3.749 | 744 | 99.703 | 752 / 100.777 |
| 128 | 2 | 478748 | 90.156 | 5.531 | 736 | 98.159 | 744 / 101.699 |

Более крупные native JPEG требуют отдельного расчёта. H100 latency не измерена.

## Запуск

Ноутбук: notebooks/disentangle_bifpn.ipynb. Выбор arm: control, w64_r1,
w64_r2, w64_r2_cross32, w128_r1, w128_r2. По умолчанию w64_r2_cross32, общий размер 704.
use_max_size=True переключает на максимум из таблицы и новое имя run.

Все YAML начинаются с disentangle_b2_li704_bifpn_. Общий контроль наследует
disentangle_fuse.yaml: 6 эпох × 24000 показов, последние 2 эпохи full-frame,
без полных проходов, стандартные pretrained RGB/JPEG веса. Пути и runtime — .env.
Сравнивать варианты сначала при одинаковом размере, forward batch, effective
batch и seed. Максимальные размеры проверяют одновременно ёмкость и разрешение.

CLI-пример:

```bash
python -m src.training --config configs/experiments/disentangle_b2_li704_bifpn_w64_r2.yaml
```

На локальной Windows NumPy импортируется перед torch для обхода конфликта MKL;
в ноутбуке этот порядок уже предусмотрен. Полное обучение не запускалось.

Проверки: tests/test_bifpn.py; полный набор тестов и GPU BF16 smoke.
Численные результаты: runs/bifpn_budget_20260915/measurements.json и
cuda_verification.json.

## BiFPN 64×2 + attention 16→32

Конфиг `disentangle_b2_li704_bifpn_w64_r2_cross32.yaml` наследует вариант
без attention и меняет только run_name и disentangle_cross_strides: [32].
Обучение и losses прежние. Сравнение: arm=w64_r2 против w64_r2_cross32,
одинаковый use_max_size, seed и batch. При use_max_size=True оба используют 752.

Полный eval FlopCounterMode, native JPEG 1080×1920:

| RGB | GFLOPs |
|---|---|
| 704 | 86.907746640 |
| 752 | 99.145330684 |
| 760 | 101.618514180 |

Максимум с шагом 8 — 752. Добавка attention при 704: 0.349904896 GFLOPs.
Результаты: runs/bifpn_budget_20260915/cross32_measurements.json.
