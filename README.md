# Эксперименты по генерации радиологических заключений

В работе сравниваются режимы `zero-shot` и `fine-tuning` для модели `Salesforce/blip2-flan-t5-xl` на двух наборах рентгеновских изображений:

- `IU X-Ray` (`X-iZhang/IU-Xray-RRG`)
- `MIMIC-CXR` (`itsanmolgupta/mimic-cxr-dataset`)

Основной сценарий воспроизведения представлен в [experiments.ipynb](experiments.ipynb). В ноутбуке выполняется запуск экспериментов, выводятся лексические метрики, строятся графики и затем проводится поведенческий анализ предсказаний модели.

В каталоге `results/` уже присутствуют сохранённые артефакты предыдущих запусков. Поэтому при режиме `auto` вычисления могут не пересчитываться заново, а извлекаться из готовых файлов. Ниже описывается текущая логика кода из `src/`.

## Постановка экспериментов

Рассматриваются четыре эксперимента:

1. `iu_xray_zero_shot`
2. `mimic_cxr_zero_shot`
3. `iu_xray_fine_tuned`
4. `mimic_cxr_fine_tuned`

Экспериментальная схема задаётся в [src/config.py](src/config.py).

- В режиме `zero-shot` модель используется без дополнительного обучения и генерирует заключения на `test`-сплите.
- В режиме `fine-tuning` выполняется дообучение на `train`, отбор лучшей эпохи по `validation` и итоговая оценка на `test`.

Параметры по умолчанию:

- модель: `Salesforce/blip2-flan-t5-xl`
- prompt:
  `Radiology findings. Write one or two short clinical sentences about visible findings only. Do not describe the image itself. Do not use phrases like 'chest x-ray', 'image of', or 'patient with'.`
- seed: `42`
- число эпох:
  `3` для `iu_xray_fine_tuned`, `2` для `mimic_cxr_fine_tuned`
- learning rate: `1e-5`
- weight decay: `0.01`
- max label length: `96`

Для `IU X-Ray` перед экспериментами выполняется балансировка train- и validation-частей по normal и abnormal случаям. Тестовая часть при этом сохраняется без изменений.

## Структура проекта

- [experiments.ipynb](experiments.ipynb) — основной ноутбук с запуском и анализом.
- [requirements.txt](requirements.txt) — зависимости Python.
- [src/config.py](src/config.py) — конфигурация модели, датасетов и списка экспериментов.
- [src/workflow.py](src/workflow.py) — общий сценарий запуска, повторного использования артефактов, анализа и визуализации.
- [src/datasets.py](src/datasets.py) — загрузка датасетов и подготовка сплитов.
- [src/evaluation.py](src/evaluation.py) — генерация предсказаний, вычисление метрик и сохранение результатов.
- [src/training.py](src/training.py) — цикл fine-tuning, валидация по эпохам, TensorBoard-логи и сохранение лучшей модели.
- [src/utils.py](src/utils.py) — вспомогательные функции.
- `results/` — сохранённые артефакты и графики.

## Вычислительная среда

Платформа автоматически определяется в [src/workflow.py](src/workflow.py):

- при наличии `CUDA` используется `cuda`
- при отсутствии `CUDA`, но при доступности Apple Metal используется `mps`
- в остальных случаях используется `cpu`

Платформозависимые параметры:

- `cuda`: `float16`, train/eval batch size `2`
- `mps`: `float16`, train/eval batch size `1`, `grad_accum_steps=8`
- `cpu`: `float32`, batch size `1`

Для первого запуска необходим доступ к сети, так как модель и датасеты загружаются из Hugging Face. Для `blip2-flan-t5-xl` также требуется заметный объём памяти и дискового пространства; запуск на `cpu` возможен, но ожидаемо медленный.

Версия Python в проекте явно не фиксируется. В репозитории отсутствуют `pyproject.toml` и `.python-version`, поэтому используется отдельное виртуальное окружение с версией Python, совместимой с [requirements.txt](requirements.txt).

## Подготовка окружения

Минимальный сценарий подготовки окружения:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install notebook ipykernel
python -m ipykernel install --user --name experiments
```

Файл `.env` считывается в [src/workflow.py](src/workflow.py). При наличии переменной `HF_ACCESS_TOKEN` выполняется попытка входа в Hugging Face.

Минимальная запись:

```dotenv
HF_ACCESS_TOKEN=hf_your_token_here
```

Если доступ к используемым ресурсам допускает анонимную загрузку, запуск возможен и без токена. При ограниченном доступе к модели или датасетам токен требуется.

## Примеры запуска

### Через ноутбук

```bash
source .venv/bin/activate
jupyter notebook experiments.ipynb
```

В ноутбуке выполняется последовательный запуск ячеек сверху вниз.

### Через Python API

Отдельный CLI в проекте не предусмотрен; запуск собирается через функции из `src`.

```python
from pathlib import Path

from src import (
    DEFAULT_RUN_MODES,
    FINE_TUNED_EXPERIMENT_NAMES,
    ZERO_SHOT_EXPERIMENT_NAMES,
    create_data_limits,
    create_runtime_config,
    initialize_runtime,
    run_experiments,
)

run_modes = dict(DEFAULT_RUN_MODES)
selected_experiments = ZERO_SHOT_EXPERIMENT_NAMES + FINE_TUNED_EXPERIMENT_NAMES

config = create_runtime_config(
    results_root_dir=Path("results"),
    data_limits=create_data_limits(
        zero_shot=None,
        train=None,
        validation=None,
        test=None,
    ),
)

state = initialize_runtime(config)
results = run_experiments(
    state,
    experiment_names=selected_experiments,
    run_modes=run_modes,
)
```

## Режимы запуска

Режимы задаются словарём `RUN_MODES` и используются внутри [src/workflow.py](src/workflow.py).

| Режим | Интерпретация |
| --- | --- |
| `auto` | Сначала ищутся готовые артефакты в `results/`; при их отсутствии выполняется новый запуск. |
| `reuse` | Допускается только использование уже сохранённых артефактов; при их отсутствии возникает ошибка. |
| `rerun` | Старые результаты игнорируются и эксперимент запускается заново. Для fine-tuning предварительно очищаются служебные артефакты запуска. |

Пример настройки:

```python
RUN_MODES = dict(DEFAULT_RUN_MODES)

RUN_MODES["iu_xray_zero_shot"] = "reuse"
RUN_MODES["mimic_cxr_zero_shot"] = "rerun"
RUN_MODES["iu_xray_fine_tuned"] = "auto"
RUN_MODES["mimic_cxr_fine_tuned"] = "rerun"
```

## Ограничение объёма данных

Для быстрого прогона используется `create_data_limits()`.

```python
config = create_runtime_config(
    results_root_dir=Path("results"),
    data_limits=create_data_limits(
        zero_shot=50,
        train=200,
        validation=50,
        test=50,
    ),
)
```

В таком варианте:

- `zero-shot` выполняется только на первых `50` объектах `test`
- `fine-tuning` использует первые `200` объектов `train`
- `validation` и `test` также усекаются

Такой режим удобен для smoke test и проверки воспроизводимости полного pipeline.

## Содержимое `experiments.ipynb`

Ноутбук организован как последовательное исследовательское повествование.

| Ячейки | Содержание | Сохраняемые артефакты |
| --- | --- | --- |
| `1` | Введение и пояснение режимов `auto` / `reuse` / `rerun`. | Нет |
| `2` | Импорт `src` и публичных функций пакета. | Нет |
| `3-4` | Формирование `RUN_MODES`, выбор экспериментов, сбор runtime-конфига, инициализация состояния через `initialize_runtime()`, вывод обнаруженной платформы. | При необходимости создаются рабочие каталоги внутри `results/<model>/` и `results/<model>/plots/` |
| `5-7` | Запуск выбранных экспериментов через `run_experiments()`, затем вывод источника результатов: `saved_artifacts`, `saved_model_evaluation` или `fresh_run`. | JSON-артефакты экспериментов, модели, TensorBoard-логи, графики длины текстов и кривые обучения |
| `8-9` | Вывод лексических метрик (`BLEU-4`, `ROUGE-L`, `METEOR`) и построение сравнительных графиков. | `plots/lexical_bleu_4.png`, `plots/lexical_rouge_l.png`, `plots/lexical_meteor.png` |
| `10-13` | Построение краткого сравнительного текстового анализа для перехода от `zero-shot` к `fine-tuning` на каждом датасете. | Нет |
| `14-15` | Поведенческий анализ fine-tuned моделей: разнообразие генерации, доля шаблонов `no acute`, полнота по патологиям, наиболее частые ответы. | Нет |
| `16-17` | Визуализация `no_acute_rate`. | `plots/behavior_no_acute_rate.png` |
| `18-19` | Визуализация `no_acute_on_abnormal_rate`. | `plots/behavior_no_acute_on_abnormal_rate.png` |
| `20-21` | Визуализация `unique_prediction_ratio`. | `plots/behavior_unique_prediction_ratio.png` |
| `22-23` | Визуализация `top5_share`. | `plots/behavior_top5_share.png` |
| `24-25` | Визуализация `any_pathology_recall`. | `plots/behavior_any_pathology_recall.png` |
| `26-27` | Визуализация полноты по отдельным патологическим терминам. | `plots/behavior_pathology_term_recall.png` |
| `28-29` | Итоговая интерпретация результатов. | Нет |

### Логика выполнения экспериментов

При запуске ноутбука фактически выполняется следующий сценарий:

1. `initialize_runtime()`:
   загружается `.env`, при необходимости выполняется логин в Hugging Face, фиксируются seed, создаются рабочие директории и определяется платформа.
2. `run_experiments()`:
   по очереди вызываются четыре эксперимента.
3. `run_zero_shot_experiment()`:
   либо считываются `zero_shot_*` артефакты из `results`, либо загружается модель и оценивается `test`.
4. `run_fine_tuned_experiment()`:
   либо используются готовые `fine_tuned_*` файлы, либо запускается обучение и последующая оценка лучшей модели.

## Fine-tuning

Логика fine-tuning реализована в [src/training.py](src/training.py).

Основные этапы:

- модель загружается через `Blip2ForConditionalGeneration.from_pretrained(...)`
- большая часть параметров замораживается, а обучаемыми остаются `language_projection`, `query_tokens` и `qformer`
- train/validation-сплиты преобразуются в входы процессора и токенизированные labels
- после каждой эпохи вычисляются `val_loss` и текстовые метрики на `validation`
- лучшая эпоха выбирается по `val_loss`
- лучшая модель сохраняется в `best_model/`
- после завершения обучения лучшая модель повторно загружается и оценивается на `test`

Дополнительно сохраняются:

- TensorBoard-логи
- `fine_tuned_training_log.jsonl`
- `fine_tuned_training_history.json`
- `best_epoch_summary.json`
- предсказания и метрики на validation после каждой эпохи

## Метрики

В [src/evaluation.py](src/evaluation.py) вычисляются:

- `BLEU-4`
- `ROUGE-1`
- `ROUGE-L`
- `METEOR`
- средняя длина предсказания
- средняя длина референса

В ноутбуке отдельно визуализируются:

- `BLEU-4`
- `ROUGE-L`
- `METEOR`

### Поведенческие показатели

Поведенческий анализ реализован в [src/workflow.py](src/workflow.py) функцией `analyze_prediction_behavior()`.

Используются следующие показатели:

- `unique_prediction_ratio` — доля уникальных предсказаний
- `top5_share` — доля выборки, покрытая пятью наиболее частыми формулировками
- `no_acute_rate` — доля отрицательных шаблонов типа `no acute`
- `no_acute_on_abnormal_rate` — доля таких ответов в случаях, где в референсе присутствует патология
- `any_pathology_recall` — частота упоминания патологии, если она есть в референсе
- `term_recall` — полнота по отдельным группам патологических терминов

По умолчанию учитываются группы терминов:

- `cardiomegaly`
- `effusion`
- `edema`
- `atelectasis`
- `pneumonia`
- `opacity`
- `congestion`
- `pneumothorax`
- `fracture`
- `emphysema`

## Артефакты запуска

Корневой путь к результатам имеет вид:

```text
results/<sanitized_model_name>/
```

Для текущей модели:

```text
results/Salesforce__blip2_flan_t5_xl/
```

### Zero-shot

```text
results/Salesforce__blip2_flan_t5_xl/<dataset>/zero_shot/
  zero_shot_predictions.json
  zero_shot_metrics.json
```

### Fine-tuning

```text
results/Salesforce__blip2_flan_t5_xl/<dataset>/fine_tuned/
  best_model/
  last_model/
  tensorboard/
  validation_by_epoch/
    epoch_1_validation_predictions.json
    epoch_1_validation_metrics.json
    ...
  fine_tuned_training_log.jsonl
  fine_tuned_training_history.json
  best_epoch_summary.json
  fine_tuned_test_predictions.json
  fine_tuned_test_metrics.json
```

### Графики

```text
results/Salesforce__blip2_flan_t5_xl/plots/
  lexical_bleu_4.png
  lexical_rouge_l.png
  lexical_meteor.png
  behavior_no_acute_rate.png
  behavior_no_acute_on_abnormal_rate.png
  behavior_unique_prediction_ratio.png
  behavior_top5_share.png
  behavior_any_pathology_recall.png
  behavior_pathology_term_recall.png
  iu_xray/
    zero_shot_lengths.png
    fine_tuned_loss_curve.png
    fine_tuned_learning_curve.png
    fine_tuned_test_lengths.png
    validation_by_epoch/
      epoch_1_validation_lengths.png
      ...
  mimic_cxr/
    zero_shot_lengths.png
    fine_tuned_loss_curve.png
    fine_tuned_learning_curve.png
    fine_tuned_test_lengths.png
    validation_by_epoch/
      epoch_1_validation_lengths.png
      ...
```

В уже сохранённом `results/` могут присутствовать и дополнительные файлы от более ранних запусков. Список выше описывает артефакты, создаваемые текущим кодом.

## Используемые датасеты

### IU X-Ray

В [src/config.py](src/config.py) задаются:

- Hugging Face dataset: `X-iZhang/IU-Xray-RRG`
- config: `impression_section`
- поле изображения: `main_image`
- референсный текст: `findings_section`

### MIMIC-CXR

В [src/config.py](src/config.py) задаются:

- Hugging Face dataset: `itsanmolgupta/mimic-cxr-dataset`
- config: `None`
- поле изображения: `image`
- референсный текст: `impression`

Если датасет не содержит готовых `train`/`validation`/`test`, в [src/datasets.py](src/datasets.py) автоматически формируются сплиты `70/15/15`.

## Замечания по воспроизведению

- `auto` не означает обязательный новый расчёт; при наличии артефактов они будут переиспользованы
- `rerun` для fine-tuning очищает `best_model/`, `last_model/`, `tensorboard/`, `validation_by_epoch/` и старые `fine_tuned_*` файлы
- в [src/evaluation.py](src/evaluation.py) генерация идёт с `do_sample=True`, `temperature=0.8`, `top_p=0.9`, поэтому при полном пересчёте возможны небольшие вариации текстов и метрик
- на `mps` включается `PYTORCH_ENABLE_MPS_FALLBACK=1`
- `matplotlib` использует кэш в `results/.matplotlib`
- графики не только отображаются в ноутбуке, но и сохраняются на диск

## Навигация по коду

- изменение модели, prompt, числа эпох и learning rate:
  [src/config.py](src/config.py)
- изменение загрузки датасетов и балансировки:
  [src/datasets.py](src/datasets.py)
- изменение генерации и вычисления метрик:
  [src/evaluation.py](src/evaluation.py)
- изменение fine-tuning:
  [src/training.py](src/training.py)
- изменение сценария экспериментов, режимов запуска и графиков:
  [src/workflow.py](src/workflow.py)
