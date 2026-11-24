# ltx-runner

Docker-образ для Vast.ai: [LTX-2.5](https://github.com/Lightricks/LTX-2) с готовыми зависимостями и мини-сервером задач.
Веса модели в образ не входят и скачиваются при старте с Hugging Face (лицензия LTX-2.x Community License, нужен `HF_TOKEN` с принятой лицензией).

Собирается GitHub Actions при каждом push в `main` и публикуется как `ecnurway/ltx-runner:latest`.

## Запуск на Vast

- Образ: `ecnurway/ltx-runner:latest`
- Порты: `-p 8189:8189 -p 8190:8190`
- Переменные: `-e RUNNER_TOKEN=… -e HF_TOKEN=… [-e LTX_VARIANTS="bf16 nvfp4"]`
- Onstart: `bash /opt/ltx-runner/start.sh`
- Нужна машина с драйвером CUDA ≥ 13.2 (окружение LTX использует torch cu132).

Содержимое `scripts/`: `runner.py` (сервер задач, порт 8190), `hwtest.py` (тест железа), `ltx_batch.py` (генерация шотов
с моделями в памяти), `fetch_weights.sh` (веса), `start.sh` (старт).
