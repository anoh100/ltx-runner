#!/bin/bash
# Старт машины из образа: сервер задач (:8190), раздача файлов (:8189), затем фоновое скачивание весов LTX-2.5.
# Переменные окружения (задаются при аренде): RUNNER_TOKEN, HF_TOKEN, опционально LTX_VARIANTS="bf16 nvfp4".
mkdir -p /root/logs /root/work
set -a; source /etc/environment 2>/dev/null; set +a
cp /opt/ltx-runner/*.py /root/work/ 2>/dev/null
nohup python3 /opt/ltx-runner/runner.py > /root/logs/runner.log 2>&1 &
(cd /root/logs && nohup python3 -m http.server 8189 > /dev/null 2>&1 &)
nohup bash /opt/ltx-runner/fetch_weights.sh > /root/logs/weights.log 2>&1 &
echo "STAGE image_started $(date)" >> /root/logs/setup.log
