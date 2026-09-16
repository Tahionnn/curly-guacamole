# AIIJC 2026: JPEG640 baseline

Проект для сегментации AI-изменённых областей изображения.
Архитектура: **PVT-v2-B2 + native JPEG branch + LocalFusion + EMCAD**.
История экспериментов сохранена в ветке `codex/emcad-baseline`.

## Структура

- `src/` — модель, данные, обучение, оценка и inference.
- `configs/baseline.yaml` — короткий бейзлайн.
- `configs/baseline_long.yaml` — длинный бейзлайн, наследующий короткий.
- `notebooks/train.ipynb` — выбор рецепта, проверка бюджета, запуск.
- `notebooks/experiments.ipynb` — просмотр результатов.
- `runs/` — checkpoints, resolved configs, метрики, OOF, protocol и notes.
- `runs/baseline` и `runs/baseline_long` — сохранённые короткий и длинный JPEG640 baseline. Старые эксперименты и аудиты перенесены в `runs/archive/20260915`; таблица исходных путей — `relocation_manifest.json` внутри архива. Исторические snapshots и checkpoint сохраняют исходные имена запусков.
- `tests/` — проверки поддерживаемого пайплайна.
- `docs/archive/` — исторические описания экспериментов.

## Окружение

Conda environment `challenges` описано в `environment.yml`.
Локальный Python: `D:\Apps\anaconda3\envs\challenges\python.exe`.
Скопируйте `.env.example` в `.env`, задайте пути и оборудование.
JPEG stem ожидает `DCT_djpeg.pth` в корне либо `model.jpeg_pretrained`.
RGB pretrained загружается через timm при новом обучении. Resume и inference
не требуют повторного скачивания pretrained.

Чтение JPEG использует CFFI/libjpeg: первый запуск собирает модуль в
`runs/.jpeg_decoder`. CUDA lookup использует Triton, CPU — PyTorch.
Если стандартный кэш недоступен, задайте `TRITON_CACHE_DIR` в доступный каталог.

Приоритет оборудования: environment → `.env` → YAML → defaults.
Пути разрешаются относительно корня проекта. Run сохраняет итоговые значения,
включая batch, accumulation, число GPU и digests data protocol.

Наблюдавшийся короткий бейзлайн: **2 GPU × batch 8, accumulation 1,
workers 10, bf16**. Defaults без `.env` остаются batch 4 / accumulation 4,
как в исходных рецептах; это другой режим обучения. Accumulation не заменяет
forward batch для BatchNorm. Multi-GPU SyncBN требует CUDA/NCCL;
Windows/Gloo его не поддерживает.

## Рецепты

| Параметр | baseline | baseline_long |
|---|---|---|
| RGB input | 640 × 640, stretch | тот же |
| Эпохи | 6 | 18 |
| Выборка | 24 000 с replacement на эпоху | 15 sampled + 3 полных прохода |
| Финальные full-frame эпохи | 2 | 3 |
| LR | warmup → cosine | warmup → constant → cosine в полных проходах |

`full_pass_epochs` — проход по каждой строке train split без replacement,
с сохранением неполного batch. Validation не добавляется в обучение.
`augmentation.final_full_frame_epochs` отключает crop независимо от sampling.

SyncBN применяется к encoder и decoder. JPEG-ветка сохраняет нормализацию
отдельно для каждого native frame; LocalFusion не содержит BN. Примеры без
JPEG-коэффициентов проходят RGB-путь. Сохранены исходные decoder auxiliary,
segmentation и classification heads.

Loss: BCE + `dice_weight` × Dice по всем изображениям + 0.3 × classifier BCE
+ `aux_weight` × (decoder BCE + `dice_weight` × decoder Dice).
Defaults: `dice_weight=1.0`, `aux_weight=0.4`.

## Настройки

| Поле | Смысл |
|---|---|
| `run_name` | Имя запуска и каталога для resume |
| `model.encoder` | timm encoder с нужными scales |
| `model.jpeg_channels` | Ширины JPEG-пирамиды |
| `model.jpeg_pretrained` | Путь к pretrained JPEG stem |
| `train.encoder_lr` | LR RGB encoder |
| `train.jpeg_lr` | LR JPEG branch и fusion |
| `train.head_lr` | LR decoder и остальных heads |
| `train.samples_per_epoch` | Число sampled rows до полных проходов |
| `train.full_pass_epochs` | Число финальных полных проходов |
| `train.grad_accum_steps` | Batches на optimizer step |
| `train.warmup_fraction` | Доля шагов warmup |
| `train.finetune_from` | Checkpoint для нового обучения; путь относительно runs |
| `train.finetune_weights` | `model` или `ema`; optimizer/scheduler/EMA создаются заново |
| `eval.selection_small_mask_weight` | Вес Dice для GT-площади (1%, 5%] при выборе checkpoint/порогов |

Сетки порогов, AdamW, EMA, AMP и clipping заданы в `src/config.py`;
YAML может переопределить поддерживаемые поля. Неизвестные ключи отклоняются.
`AIIJC_ACCUM_STEPS` сохранён как имя environment override для `grad_accum_steps`.

## Запуск

После активации `challenges`, из корня проекта:

```powershell
python -m src.training --config configs/baseline.yaml
python -m src.training --config configs/baseline_long.yaml
```

`resume: true` продолжает run при наличии `ckpt/last.pt`.
Для независимого эксперимента задавайте новое `run_name`.
Checkpoint сохраняется после training-фазы; сбой validation/report не требует
повторять уже обученную эпоху. Resume проверяет protocol, архитектуру, loss,
аугментации, расписание, batch/accumulation и число GPU. EMA сохраняет счётчик.

## Оценка и submission

Development validation восстанавливает вероятности до исходного размера GT
до бинаризации. Weighted selection score отличается от обычного AIC;
отчёты сохраняют оба. Пороги берутся из выбранного checkpoint/run.

```powershell
python -m src.inference runs/baseline submissions/baseline
```

Постобработка поддерживает `area_cap`: если вероятность изменения кадра ниже
`cls_threshold`, порог маски повышается до площади **строго меньше** cap.
При `area_cap=0` такой кадр обнуляется, как прежде. При cap ≤ 0.01 остаток
маски не считается ложной тревогой, но может сохранить часть Dice позитивного
кадра. Правило общее для подбора порогов, отчётов и inference.

Переподбор по сохранённым development-гистограммам, без модели и GPU:

```powershell
python -m src.tools.retune --run runs/baseline --small-mask-weight 1
```

По умолчанию команда только печатает результат и команду для submission.
`--small-mask-weight 1` выбирает по официальному AIC; без флага используется
вес из гистограмм. `--write` атомарно сохраняет `best`, `val_best`, `best_aic`
и происхождение новой точки в `summary.json`. Файл весов и старые development-
отчёты не меняются. Inference проверяет хеш checkpoint; holdout дополнительно
проверяет хеш гистограмм. Запись разрешена после завершения обучения и до
начала holdout. Подбор по holdout не поддерживается.

Для нового обучения с сеткой cap есть
`configs/experiments/baseline_capped_gate.yaml`. Обычные конфиги сохраняют
`eval.area_caps: [0.0]`. `Run.operating_point()` и
`ValidationResult.operating_point` возвращают четыре значения:
`(mask_threshold, cls_threshold, min_area, area_cap)`.

Inference принимает `--batch-size`, `--workers` и `--post-workers` (по умолчанию 8;
0 отключает потоки постобработки). Очереди масок и записи PNG ограничены.
Явные `--mask-threshold`, `--cls-threshold`, `--min-area` можно дополнить
`--area-cap` и `--n-bins`; без `--n-bins` сетка берётся из snapshot запуска.

Команда записывает `submission.csv` и одноканальные PNG 0/255 в `predictions/`.
Перед отправкой упакуйте эти два элемента в ZIP согласно `AIIJC_RULES.md`.
Holdout — отдельная финальная оценка с замороженными порогами:

```powershell
python -m src.eval --run runs/baseline
```

После holdout claim run нельзя продолжать обучать или использовать для finetune.
Protocol: `runs/validation_protocol_20260908/protocol`. Protocol и результаты
не входят в обычный Git checkout: сохраняйте их отдельно. Тестовые изображения
используются только для финальных predictions.

## Производительность и проверки

```powershell
python -m src.training.profiling --config configs/baseline.yaml --profile-data
python -m src.training.profiling --config configs/baseline.yaml --workers 2 4 10
python -m pytest tests -q
```

Profiler использует disposable model без pretrained/checkpoints и сравнивает
loader с GPU replay; competition run не сохраняется.
`count_gflops` считает полный eval-forward через `FlopCounterMode` и требует
`native_size`: стоимость JPEG зависит от исходного размера.
Оценка для native 1024 × 1024 не гарантирует лимит для остальных размеров.
Лимиты: ≤100 GFLOPs и ≤50 ms/image на H100; latency измеряется отдельно.

## История

Новая версия snapshots — `jpeg640_v1`. `SnapshotAdapter` читает старые
`emcad_v1` snapshots поддерживаемой JPEG/local/SyncBN архитектуры.
State dict keys сохранены. Остальные старые архитектуры запускаются
из `codex/emcad-baseline`. Run directories, checkpoints, notes, OOF и protocol
не переименованы и не удалены.

Инварианты чистки: `docs/superpowers/specs/2026-09-15-jpeg640-baseline-design.md`.
