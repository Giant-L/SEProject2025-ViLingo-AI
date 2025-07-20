# =========================================================================
#  阶段一：构建器 (The Builder)
#  在这个阶段，我们安装所有依赖并下载模型。
# =========================================================================
FROM python:3.10 as builder

# 设置工作目录
WORKDIR /app

# 复制生产依赖列表文件
COPY requirements.txt requirements.txt

# 设置缓存目录
ENV HF_HOME=/app/cache/huggingface
ENV TORCH_HOME=/app/cache/torch
ENV XDG_CACHE_HOME=/app/cache

# 安装Python依赖
# 使用镜像源可以加速
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制应用代码
COPY ./src /app/src

# 在构建时预热并下载所有模型到指定的缓存目录
RUN python -c "from src.vilingo.core import models; print('开始在构建器中预下载所有模型...'); models.load_all_models(); print('所有模型已在构建器中成功下载并缓存。')"


# =========================================================================
#  阶段二：最终镜像 (The Final Image)
#  这个阶段只包含运行应用所必需的东西。
# =========================================================================
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# --- 从构建器(builder)阶段复制所需的文件 ---

# 1. 复制已安装的Python库
#    找到builder阶段中Python库存放的路径并复制过来
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# 2. 复制已下载的AI模型缓存
COPY --from=builder /app/cache /root/.cache

# 3. 复制我们的应用代码
COPY --from=builder /app/src /app/src

# --- 设置运行环境 ---
# 再次设置环境变量，让程序在运行时能找到模型缓存
ENV HF_HOME=/root/.cache/huggingface
ENV TORCH_HOME=/root/.cache/torch
ENV XDG_CACHE_HOME=/root/.cache

# 暴露端口
EXPOSE 8000

# 设置启动命令
CMD ["uvicorn", "src.vilingo.main:app", "--host", "0.0.0.0", "--port", "8000"]