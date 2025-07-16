# 使用一个官方的、轻量的Python 3.10镜像作为基础
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖，ffmpeg用于处理音视频
# apt-get clean && rm -rf /var/lib/apt/lists/* 用于减小镜像体积
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 复制生产依赖列表文件
COPY requirements.txt requirements.txt

# 设置Hugging Face, PyTorch, Whisper的缓存目录，模型将被下载到镜像的这些位置
ENV HF_HOME=/root/.cache/huggingface
ENV TORCH_HOME=/root/.cache/torch
ENV XDG_CACHE_HOME=/root/.cache

# 安装Python依赖。--no-cache-dir 减小镜像体积
RUN pip install --no-cache-dir -r requirements.txt

# 将我们的核心应用代码复制到镜像中
COPY ./src /app/src

#  在构建时预热并下载所有模型 ---
#
#  此命令会执行模型加载脚本，这将触发所有AI模型的下载。
#  模型文件会被保存在上面ENV设置的缓存目录中，并成为Docker镜像的一部分。
#  这一步会显著增加构建时间和最终镜像的大小，但这是我们所期望的。
#
RUN python -c "from src.vilingo.core import models; print('开始预下载所有模型...'); models.load_all_models(); print('所有模型已成功下载并缓存。')"


# 暴露端口，让外部可以访问FastAPI服务
EXPOSE 8000

# 容器启动时执行的命令
# 它会启动uvicorn服务器来运行位于 src/vilingo/main.py 文件中的 app 对象
CMD ["uvicorn", "src.vilingo.main:app", "--host", "0.0.0.0", "--port", "8000"]