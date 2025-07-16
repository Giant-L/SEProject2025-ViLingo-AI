# src/vilingo/core/models.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import whisper

# ...

# --- 模型名称和设备配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Whisper 模型
WHISPER_MODEL = "small"  # <--- 在这里定义了 Whisper 模型版本

# 2. 文本概括 LLM 模型
SUMMARIZER_MODEL = "Qwen/Qwen2-7B-Instruct"  # <--- 在这里定义了 Qwen2 模型

# 3. 语义相似度模型
SIMILARITY_MODEL = 'all-MiniLM-L6-v2'  # <--- 在这里定义了 all-MiniLM-L6-v2 模型

CACHE_DIR = "/root/.cache" # Docker镜像内的缓存目录


def load_all_models():
    """
    这个函数会使用上面的变量来告诉具体的库应该下载哪个模型。
    """
    print(f"正在使用设备: {DEVICE}")

    # 加载 Whisper Model
    # whisper.load_model 函数会读取 WHISPER_MODEL 变量
    whisper_model = whisper.load_model(WHISPER_MODEL, device=DEVICE, download_root=f"{CACHE_DIR}/whisper")
    MODEL_REGISTRY['whisper'] = whisper_model

    # 加载 Sentence Transformer Model
    # SentenceTransformer 类会读取 SIMILARITY_MODEL 变量
    st_model = SentenceTransformer(SIMILARITY_MODEL, device=DEVICE, cache_folder=CACHE_DIR)
    MODEL_REGISTRY['sentence_transformer'] = st_model

    # 加载 Qwen2-7B-Instruct Model
    # from_pretrained 函数会读取 SUMMARIZER_MODEL 变量
    qwen_model = AutoModelForCausalLM.from_pretrained(
        SUMMARIZER_MODEL,
        # ... 其他参数
    )
    # ...