# AIIJC 2026 · AIC — финальное решение: `disentangle_b2_li760_r8_all_data_hard_pixel_ft`

Сегментация области изображения, изменённой ИИ-редактором, по одному JPEG-кадру.
Метрика соревнования — **AIC Score**: гармоническое среднее `Dice_pos` (по
позитивным примерам) и `(1 - FPR_neg)` (по негативным, порог ложной тревоги —
предсказанная площадь ≥ 1% кадра). Жёсткие лимиты: **≤ 100 GFLOPs** на
изображение (`torch.utils.flop_counter.FlopCounterMode`, форвард модели без
DCT-препроцессинга/восстановления маски) и **≤ 50 мс** на одном H100 80 GB.
Полные правила — `AIIJC_RULES.md`.

Финальное решение команды — **`disentangle_b2_li760_r8_all_data_hard_pixel_ft`**:
PVT-v2-B2-li энкодер + native-JPEG forensic-ветка + DG-Force disentangle
(канальные/граничные бутылочные горлышки с cross-attention 16→32) + EMCAD-
декодер, обученный в три стадии (базовое обучение → дотюн на всех размеченных
данных → дотюн с hard-pixel loss). Полная воспроизводимая цепочка — `solution.ipynb`.

## Железо и версии

| | |
|---|---|
| Обучение (референс проекта) | 1× H100 80 GB (лимиты соревнования также заданы под H100 80 GB, RAM 1.48 TB, CPU Xeon Platinum 8358 @2.6 GHz, 127 ядер) |
| Проверялось локально на | RTX 3070 8 GB (только smoke-тесты отдельных стадий/модулей, не полное обучение — см. `configs/experiments/*.md`) |
| Python | 3.11 (см. `environment.yml`) |
| Фреймворк | PyTorch + `torchvision`, `timm` (PVT-v2 энкодеры), `albumentations`, `opencv-python-headless`, CUDA-путь JPEG-lookup — `triton` (Linux) / `triton-windows` (Windows); на CPU — эквивалентный fallback на чистом PyTorch |
| Версии пакетов | не закреплены жёстко в `environment.yml` (соответствует политике проекта: последние совместимые версии, не пины) — `solution.ipynb` ставит тот же набор через `pip` при каждом запуске |

**TODO (авторам):** впишите точные версии `torch`/`timm`/CUDA/драйвера, на которых
получен финальный сабмит (`pip freeze` после обучения), если для точной
побитовой воспроизводимости на другой машине это важно.

## Структура репозитория

```
solution.ipynb          # единственный источник правды: установка зависимостей →
                        # данные → 3 стадии обучения финального конфига → инференс → submission.zip
notebooks/
  eda.ipynb             # домены (включая скрытый plain_l8/plain_l9), qt_luma, площади масок, train/test
  model_comparison.ipynb# сравнение архитектур по runs/*/notes.md + живой замер GFLOPs по цепочке
  ablations.ipynb       # факторная гигиена цепочки конфигов, GFLOPs-вклад модулей,
                        # абляция channel_gate, таксономия отказов, диагностика hard-pixel loss
src/
  config.py             # строгая валидация YAML (ExperimentConfig/ModelConfig/...); лишний ключ — ошибка
  modules/               # forensic_fusion.py, forensic_disentangle.py (DG-Force), jpeg_branch.py, gate_head.py, ...
  decoders/              # мини-реестр декодеров (сейчас — EMCAD)
  data/                  # датасет, аугментации, DCT-препроцессинг
  training/              # engine.run_experiment / builders / sampling / EMA / resume
  eval/                  # протокол train/development/holdout, метрики, отчёты
  inference/             # predict.py, submission.py — python -m src.inference
  forensic/              # чтение JPEG-метаданных (qtable) и коэффициентов
  tools/                 # retune.py (переподбор порогов без GPU), report_gates.py
configs/
  baseline.yaml, baseline_long.yaml   # два самодостаточных базовых рецепта (закреплено tests/test_baseline_cleanup.py)
  experiments/                         # цепочка предков финального конфига + карточки экспериментов,
                                       # которые всё ещё используются тестами (см. ниже)
runs/
  baseline/                            # текущий (jpeg640_v1) базовый прогон — notes.md с leaderboard_score
  archive/                             # исторические прогоны линии emcad_v1 (другой код, ветка codex/emcad-baseline)
  validation_protocol_20260908/        # неизменяемый протокол train/development/holdout (обязателен)
tests/                                 # pytest на синтетике — без скачивания весов и без датасета
docs/validation_protocol.md            # как устроен фиксированный протокол валидации
AIIJC_RULES.md                         # правила соревнования (метрика, лимиты, формат сабмита)
```

`configs/experiments/` после чистки хранит **только** предков финального конфига
и файлы, без которых упадут тесты (`tests/test_b2_legacy_optimizations.py`,
`tests/test_dgforce.py`, `tests/test_boundary_band_loss.py`,
`tests/test_forensic_disentangle.py::ARMS`) — это проверено запуском
`pytest tests -q` после удаления остальных 21 файлов серии `disentangle_b2_li7*`
экспериментов (bifpn-сетка, li640/704/728/736/744, jpeg_ft, plain_nonunit_ft,
reference_ft, jpeg_similarity, baseline_capped_gate) и легаси-карты
`docs/archive/`. Все 19 старых experiment-ноутбуков `notebooks/*.ipynb` удалены —
их код вошёл в `solution.ipynb` (три стадии обучения) и в новые аналитические
ноутбуки. `runs/` не тронут: чекпоинты и метрики в нём — тяжёлые артефакты и
не входят в git (`.gitignore`), но `notes.md` во всех подкаталогах отслеживаются
и остались нетронутыми для сравнения моделей.

## Как воспроизвести результат

1. Клонируйте репозиторий, откройте `solution.ipynb` (Jupyter/JupyterLab/VS Code).
2. Выполните ячейки по порядку **сверху вниз**:
   - установка зависимостей (`pip`, прямо в ноутбуке — conda-окружение не требуется);
   - определение корня проекта, создание `.env` из `.env.example`;
   - разместите данные соревнования так, чтобы получилось
     `<AIIJC_DATA_PATH>/train_stage1/stage1/train.csv` и
     `<AIIJC_DATA_PATH>/test_stage1/test_stage1/test.csv` (`AIIJC_RULES.md`);
     протокол валидации уже зафиксирован в `runs/validation_protocol_20260908/`
     и пересчитывать его не нужно;
   - положите `DCT_djpeg.pth` (предобученные веса JPEG-ветки) в корень
     репозитория — нужен только для стадии A (обучение с нуля); стадии B и C
     дотюнятся от EMA предыдущего чекпоинта и этот файл не трогают;
   - проверка бюджета GFLOPs (обязана пройти **до** обучения);
   - стадия A: `disentangle_b2_li760_r8_long` — 18 эпох (15 sampled + 3 полных
     прохода), обучение с нуля, единственная стадия с development-валидацией;
   - стадия B: `disentangle_b2_li760_r8_all_data_ft` — дотюн 3×3 прохода на
     train+development+holdout от EMA стадии A;
   - копирование чекпоинта стадии B в `runs/disentangle_b2_li760_r8_all_data_ft_ep3`
     (финальный конфиг ссылается на эту фиксированную копию, а не на «живой»
     каталог стадии B — так параллельные дальнейшие дотюны от той же точки не
     конфликтуют между собой через `resume`);
   - стадия C (финал): `disentangle_b2_li760_r8_all_data_hard_pixel_ft` — ещё
     3×3 прохода с hard-pixel loss, Triton backward JPEG-ветки, foreach-
     нормализацией градиентов;
   - инференс на test и сборка `submission.csv` + `predictions/*.png`;
   - упаковка в ZIP (`submission.csv` и `predictions/` — в корне архива, без
     вложенной папки, как требует `AIIJC_RULES.md`).
3. (Опционально, отдельно и осознанно — раздел 9 `solution.ipynb`) итоговый
   holdout: `python -m src.eval --run runs/disentangle_b2_li760_r8_all_data_hard_pixel_ft`.
   Это **разовая необратимая** проверка: после неё run нельзя продолжать
   обучать/дотюнить.

Повторный запуск любой обучающей ячейки **продолжает** сохранённый прогон
(`train.resume: true`), а не начинает заново — прерывать многочасовое обучение
и возвращаться к нему безопасно.

### Пайплайн обучения одним взглядом

| Стадия | run_name | Эпох | Старт | LR (enc/jpeg/head) | Валидация |
|---|---|---|---|---|---|
| A | `disentangle_b2_li760_r8_long` | 18 (15 + 3 full-pass) | pretrained (timm RGB + `DCT_djpeg.pth`) | 1e-4 / 3e-4 / 3e-4 | да, development, вес малых масок 1.6 |
| B | `disentangle_b2_li760_r8_all_data_ft` | 3 (все full-pass) | EMA A (`best.pt`) | 1e-5 / 3e-5 / 3e-5 | нет (`train_all_data=true`) |
| C (финал) | `disentangle_b2_li760_r8_all_data_hard_pixel_ft` | 3 (все full-pass) | EMA снапшота B | 1e-5 / 3e-5 / 3e-5 (`scheduler: none`) | нет |

Полная цепочка наследования конфигов (`extends`), архитектурные решения на
каждом звене и живой замер GFLOPs по этой цепочке — `notebooks/ablations.ipynb`,
разделы 1–2; тот же расклад в прозе — `configs/experiments/disentangle.md`.

## Инференс без переобучения

У стадий B/C нет собственной валидации, поэтому у финального run нет
собственных порогов бинаризации — они берутся с чекпоинта стадии A
(единственной стадии с development-оценкой), а веса — с финального чекпоинта
(`ckpt/last.pt`, EMA). `solution.ipynb`, раздел 8, делает это явно. Через CLI
эквивалент (после того как оба run обучены) — с ручными порогами:

```bash
python -m src.inference runs/disentangle_b2_li760_r8_all_data_hard_pixel_ft \
    submissions/disentangle_b2_li760_r8_all_data_hard_pixel_ft \
    --checkpoint last.pt \
    --mask-threshold <из runs/disentangle_b2_li760_r8_long/summary.json:best.mask_threshold> \
    --cls-threshold <best.cls_threshold> \
    --min-area <best.min_area>
```

Постобработка (`area_cap`, переподбор порогов без GPU через
`python -m src.tools.retune --run <run>`) описана в docstring
`src/inference/submission.py` и `src/tools/retune.py` — в финальном решении
используется вариант по умолчанию (`area_cap=0.0`), отдельного грид-серча по
`area_cap` для этой линии не проводилось.

## Ожидаемые числа

| | AIC (development, стадия A) | Dice_pos | FPR_neg | GFLOPs (native 1080×1920, RGB 760) |
|---|---|---|---|---|
| Текущий короткий baseline (`runs/baseline`, `jpeg640_v1`, 640, 6 эпох) | — (см. `runs/baseline/notes.md`: leaderboard AIC **0.917041**) | — | — | 92.377 (при 640) |
| Финальная архитектура (arm M, reduction=8, 760) — **без обучения**, только бюджет | — | — | — | **99.705** (проверено `count_gflops`, совпадает с комментарием в `disentangle_b2_li760_r8_long.yaml`) |
| Финал `disentangle_b2_li760_r8_all_data_hard_pixel_ft` | TODO | TODO | TODO | тот же 99.705 (архитектура не меняется в стадиях B/C) |

99.705 GFLOPs и 92.377 GFLOPs выше — не переписаны из документации, а
пересчитаны заново в этом репозитории (`notebooks/ablations.ipynb`,
`notebooks/model_comparison.ipynb`) через `count_gflops`/`FlopCounterMode`.
Латентность на H100 (лимит ≤ 50 мс) **не измерена** ни локально, ни в этом
репозитории — измеряется отдельно на целевом железе, `FlopCounterMode` считает
только число операций, не время.

**TODO (авторам):** после реального прогона стадии A и holdout-проверки
финального чекпоинта — впишите фактические `AIC`/`Dice_pos`/`FPR_neg` из
`runs/disentangle_b2_li760_r8_long/summary.json` (`best`) и, если холдаут-
проверка выполнена, из `holdout_claim.json` финального run, плюс фактический
результат на публичном/приватном лидерборде.

## Ограничения и то, что честно не проверено в этом решении

- Стадии B/C — «слепой» дотюн на train+development+holdout: осознанный трейд-
  офф «дообучить на всех размеченных данных перед сдачей» ценой отсутствия
  контроля переобучения на этих двух последних стадиях.
- Прирост от `hard_pixel_weight` не измерялся отдельным контролируемым A/B —
  стадия C меняет loss и не имеет парного контрольного прогона (см.
  `notebooks/ablations.ipynb`, раздел 6).
- `disentangle_b2_li760_long` — не однофакторный шаг цепочки (одновременно
  энкодер, разрешение и расписание обучения) — зафиксировано и объяснено в
  `notebooks/ablations.ipynb`, раздел 1, а не скрыто.
- Сиды (`random`/`numpy`/`torch`) фиксируются автоматически (`train.seed=42`,
  `src/training/base.py::set_random_seed`, вызывается в начале каждой стадии) —
  но `torch.use_deterministic_algorithms` не включён (`cudnn.benchmark=True`
  по умолчанию), то есть воспроизводимость на уровне чисел после запятой не
  гарантирована, только на уровне решения в целом.

## Тесты и проверки качества кода

```bash
python -m pytest tests -q
```

552 теста на синтетических тензорах (без скачивания весов, без датасета
соревнования; проверено `pytest --collect-only` после чистки репозитория).
На машине без `DCT_djpeg.pth` и без GPU 2 теста падают по независящим от этого
решения причинам (`test_builders.py::test_build_model_uses_model_config[True]`
требует локальный файл `DCT_djpeg.pth`; `test_config.py::test_load_baseline_config`
чувствителен к обработке Windows-путей вида `D:/...` конкретно на Linux) —
остальные проходят или помечены `skip` (аппаратные тесты без GPU). Покрывают:
строгую валидацию конфигов и их наследование,
DG-Force/disentangle модуль (включая cross-attention и bit-identical
untrained-арм гарантию), JPEG-ветку (lookup, квантование, Triton/CPU fallback),
loss-компоненты (hard-pixel, boundary band, DG patch/edge), resume/finetune
(включая «слепые» all-data эпохи), EMA, синхронный BatchNorm, распределённое
обучение и валидацию, протокол train/development/holdout и его защиту от
утечки групп, инференс и постобработку (`area_cap`, retune без GPU).

## Дополнительные ноутбуки

- **`notebooks/eda.ipynb`** — работает без GPU: домены (в т.ч. скрытая склейка
  `plain_l8`/`plain_l9` в одном ярлыке `plain`, реально подтверждено на
  протоколе — у `plain_l9` **0 негативов** из 23 955 строк), `qt_luma` из
  заголовка JPEG как бесплатный (0 GFLOPs) диагностический признак, площади
  масок по доменам, сравнение train/test по весу файла, потолок разметки при
  ресайзе (round-trip Dice).
- **`notebooks/model_comparison.ipynb`** — таблица всех `leaderboard_score` из
  `runs/*/notes.md` (реальные числа, ничего не придумано), с явной пометкой
  линии кода (`jpeg640_v1` — текущая, `emcad_v1` — архивная, другая ветка), плюс
  живой замер GFLOPs baseline → DG-Force (arm D) → cross-attention (arm M) →
  760/reduction=8 (финальная архитектура).
- **`notebooks/ablations.ipynb`** — факторная гигиена всей цепочки конфигов
  (программный дифф, не текст), GFLOPs-стоимость `supervision` (0) vs `fuse`
  режима DG-Force, абляция `channel_gate` (DG-Force и JPEG-fusion) на готовом
  чекпоинте, каркас линейных зондов, таксономия отказов по `per_image.parquet`,
  диагностика hard-pixel loss по логам стадии C.

Все три ноутбука написаны так, чтобы **не падать** при отсутствии данных/
чекпоинтов — печатают понятное сообщение о том, чего не хватает, вместо
исключения, и помечены `TODO`-блоками там, где нужна фактическая цифра с
реального прогона.

## Известные грабли (см. также `configs/experiments/*.md`)

- CUDA Turing (sm_75, например RTX 3070) не имеет аппаратного bf16 — используйте
  `AIIJC_AMP=fp16` на такой карте, `bf16` — на Ampere/Hopper и новее.
  `torch.cuda.is_bf16_supported()` — быстрая проверка.
- `DCT_djpeg.pth` нужен только для стадии A; при перезапуске стадий B/C с нуля
  (не resume, новый `run_name`) без правильного `finetune_from` обучение
  стартует с случайных весов JPEG-ветки — тесты (`test_pretrained_resume.py`,
  `test_checkpoint_finetune.py`) проверяют именно этот путь, но конфиг не
  спасёт от неправильно указанного вручную `run_name`.
- Протокол валидации неизменяем и обязателен — отключить его нельзя
  (`dataset.protocol_path` можно только *переместить*, не выключить).
- `summary.json` должен лежать рядом с чекпоинтом в `runs/<run>/` — иначе
  проверки в аналитических ноутбуках и `src/inference/submission.py`
  промолчат/упадут, а не тихо продолжат со старыми данными.
