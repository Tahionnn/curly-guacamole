# AIIJC 2026 · AIC — финальное решение: `disentangle_b2_li760_r8_all_data_hard_pixel_ft`


**Финальное решение команды**:
[PVT-v2-B2-li](https://arxiv.org/html/2106.13797v7) энкодер + [DCT-ветка CAT-Net](https://arxiv.org/abs/2511.10935) + [DG-Force disentangle](https://eccv.ecva.net/virtual/2026/poster/4416)
(канальные/граничные бутылочные горлышки с cross-attention 16→32) + [EMCAD-
декодер](https://arxiv.org/abs/2405.06880), обученный в три стадии (базовое обучение → дотюн на всех размеченных
данных → дотюн с hard-pixel loss). Полная воспроизводимая цепочка — `solution.ipynb`.


## Структура репозитория

```
solution.ipynb          # ноутбук для запуска пайплайна обучения
                        
notebooks/
  eda.ipynb             # EDA датасета
  model_comparison.ipynb# сравнение архитектур
  ablations.ipynb       # абляция
src/
  config.py             # Валидация YAML конфигов
  modules/               # модули сети
  decoders/              # реестр декодеров
  data/                  # датасет, аугментации, DCT-препроцессинг
  training/              # код трейна
  eval/                  # код валидации
  inference/             # код инференса
  forensic/              # чтение JPEG-метаданных (qtable) и коэффициентов
  tools/                 # retune.py (переподбор порогов без GPU), report_gates.py
configs/
  baseline.yaml, baseline_long.yaml   # два самодостаточных базовых рецепта
  experiments/                         # цепочка предков финального конфига + карточки экспериментов,
runs/
  baseline/                            # базовый прогон
  archive/                             # история экспериментов
  validation_protocol_20260908/        # неизменяемый протокол train/development/holdout
tests/                                 # тесты
docs/validation_protocol.md            # как устроен фиксированный протокол валидации
```


## Как воспроизвести результат

1. Создайте conda-окружение:
```bash
conda env create -f environment.yml
```
2. Создайте `.env` по примеру из .env.example 
3. Выполните ячейки `solution.ipynb`:
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
   - упаковка в ZIP.


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