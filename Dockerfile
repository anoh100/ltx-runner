# Образ для Vast.ai: LTX-2.5 (официальные пайплайны Lightricks) с готовыми зависимостями + наш сервер задач.
# Веса (~70 ГБ) в образ не входят: скачиваются при старте машины с Hugging Face (нужен HF_TOKEN с принятой лицензией).
# База — тот же образ PyTorch, что уже закэширован на многих хостах Vast, поэтому качается только наш слой.
FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime

ARG LTX_COMMIT=daa6b16fc7f0f1628f2c9e36a303e3312b333cc6
ENV DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1 UV_LINK_MODE=copy

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends git ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install -q --no-cache-dir uv "huggingface_hub[hf_xet]"

# LTX-2 с зафиксированной версией и всеми зависимостями (≈7 ГБ виртуальное окружение)
RUN git clone https://github.com/Lightricks/LTX-2 /root/LTX-2 \
    && cd /root/LTX-2 && git checkout ${LTX_COMMIT} \
    && uv sync --extra natten \
    && uv cache clean \
    && rm -rf /root/.cache

# Наши скрипты: сервер задач, тест железа, генерация пачкой, скачивание весов, старт
COPY scripts/ /opt/ltx-runner/
RUN chmod +x /opt/ltx-runner/*.sh && mkdir -p /root/work /root/logs

# Vast выполняет onstart-скрипт; в шаблоне указывать: bash /opt/ltx-runner/start.sh
CMD ["bash", "/opt/ltx-runner/start.sh"]
