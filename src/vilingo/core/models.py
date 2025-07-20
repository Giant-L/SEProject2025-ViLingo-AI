# src/vilingo/core/models.py (瘦身版)

import torch
from sentence_transformers import SentenceTransformer
import whisper
import os

MODEL_REGISTRY = {}

def get_device():
    """检测可用的最佳计算设备 (NVIDIA GPU, Apple GPU, or CPU)"""
    if torch.cuda.is_available():
        print("检测到 NVIDIA CUDA GPU，将使用 cuda 设备。")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("检测到 Apple Silicon GPU (M1/M2/M3)，将使用 mps 设备。")
        return "mps"
    else:
        print("未检测到可用GPU，将使用 CPU。")
        return "cpu"

DEVICE = get_device()
WHISPER_MODEL = "small"
SIMILARITY_MODEL = 'all-MiniLM-L6-v2'
CACHE_DIR = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))

def load_all_models():
    """
    加载运行所需的所有模型 (已移除Phi-3-mini)。
    """
    global MODEL_REGISTRY
    print(f"--- 开始加载所有AI模型到设备: {DEVICE} ---")

    try:
        # --- 加载 Whisper Model ---
        print(f"正在加载 Whisper model: {WHISPER_MODEL}...")
        whisper_model = whisper.load_model(
            WHISPER_MODEL, 
            device=DEVICE, 
            download_root=os.path.join(CACHE_DIR, "whisper")
        )
        MODEL_REGISTRY['whisper'] = whisper_model
        print("✅ Whisper model 加载完毕。")

        # --- 加载 Sentence Transformer Model ---
        print(f"正在加载 Sentence Transformer model: {SIMILARITY_MODEL}...")
        st_model = SentenceTransformer(
            SIMILARITY_MODEL, 
            device=DEVICE, 
            cache_folder=CACHE_DIR
        )
        MODEL_REGISTRY['sentence_transformer'] = st_model
        print("✅ Sentence Transformer model 加载完毕。")
        
        print("--- 所有模型均已成功加载并准备就绪！ ---")

    except Exception as e:
        print(f"❌ 模型加载过程中发生严重错误: {e}")
        raise e