# =========================================================================
#  阶段一：构建器 (The Builder)
# =========================================================================
FROM python:3.10 AS builder

# 1. 安装系统依赖 (官方源)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    default-jre-headless \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY ./src/vilingo/core/models.py /app/src/vilingo/core/models.py
RUN mkdir -p /app/src/vilingo/core && \
    touch /app/src/vilingo/__init__.py /app/src/vilingo/core/__init__.py

ENV HF_HOME=/app/cache/huggingface
ENV TORCH_HOME=/app/cache/torch
ENV XDG_CACHE_HOME=/app/cache
RUN python -c "from src.vilingo.core import models; print('开始在构建器中预下载所有模型...'); models.load_all_models(); print('所有模型已在构建器中成功下载并缓存。')"

COPY ./src /app/src

# =========================================================================
#  阶段二：最终镜像 (The Final Image)
# =========================================================================
FROM python:3.10-slim AS final

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    default-jre-headless \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local /usr/local
COPY --from=builder /app/cache /root/.cache
COPY --from=builder /app/src /app/src

ENV HF_HOME=/root/.cache/huggingface
ENV TORCH_HOME=/root/.cache/torch
ENV XDG_CACHE_HOME=/root/.cache

EXPOSE 8000

CMD ["uvicorn", "src.vilingo.main:app", "--host", "0.0.0.0", "--port", "8000"]
