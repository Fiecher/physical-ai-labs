# Help & Process Notes

## Настройка окружения в Google Colab

Если при запуске возникает ошибка импорта `cannot import name 'registry' from 'mujoco_playground'` или конфликт CUDA-плагинов JAX:

1. Установить релизную ветку `0.2.0` (до рефакторинга `1.0.0`, где изменилась структура модулей):
   ```python
   %pip uninstall -y -q playground
   %pip install -q "playground[learning] @ git+https://github.com/google-deepmind/mujoco_playground.git@v0.2.0"
   %pip install -q rsl-rl-lib tensorboard
   ```
2. Устранить конфликт `jax-cuda13-plugin` и `jax-cuda12-plugin`:
   ```python
   %pip uninstall -y -q jax-cuda13-plugin jax-cuda13-pjrt
   %pip install -q -U --force-reinstall "jax[cuda12]"
   ```
3. Перезапустить сеанс Colab (**Runtime -> Restart session**) перед запуском обучения.

## Проверка параметров среды и алгоритма

```python
# Вывод всех весов наград среды PandaPickCube
print(env_cfg.reward_config.scales)

# Вывод параметров PPO (rsl_rl)
print(manipulation_params.rsl_rl_config("PandaPickCube").to_dict())
print(f"learning_rate={cfg_dict['algorithm']['learning_rate']}")
```

> **Важно:** все изменения в `env_cfg.reward_config.scales` необходимо применять **до** вызова `raw_env = registry.load(ENV_NAME, config=env_cfg)`, иначе среда инициализируется со значениями по умолчанию.

## Интерпретация кривых обучения

| Картина                                      | Интерпретация                                                                             |
| -------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Плато на высоком уровне, SR ≈ 8–10/10        | Политика научилась хорошо выполнять задачу и стабилизировалась (`baseline`, `custom_exp`) |
| Продление обучения до 2000 итераций          | Подтверждает выход на устойчивое плато без дальнейшего роста (`scale_iters_2000`)         |
| Высокий `mean_reward` при низком `SR` (0/10) | Дисбаланс весов (reward hacking): агент максимизирует позу суставов (`qpos_strong`)       |
| Резкий рост `mean_reward` с высоким шумом    | Большой масштаб награды за целевую позицию куба (`box_target_strong`)                     |

## Инференс и запись видео

Запуск оценки обученного чекпоинта на 10 эпизодах с записью видео:

```bash
python3 inference.py --checkpoint logs/baseline/model_1000.pt \
    --record logs/baseline/video.mp4 --episodes 10
```

## Экспорт графиков из логов TensorBoard

Скрипт выгрузки кривых `Train/mean_reward` из `events.out.tfevents*` в директорию `plots/`:

```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
# Считывает скаляры Train/mean_reward, применяет сглаживание (0.6) и сохраняет .png в plots/
```
