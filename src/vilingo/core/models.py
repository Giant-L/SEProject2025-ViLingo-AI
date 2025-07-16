# src/vilingo/core/models.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import whisper
import os

# --- 1. 新增：一个字典，作为简单的模型注册表，用于存储加载好的模型 ---
MODEL_REGISTRY = {}

# --- 2. 新增：我们讨论过的、兼容 M2 Mac 的设备检测逻辑 ---
def get_device():
    """检测可用的最佳计算设备 (NVIDIA GPU, Apple GPU, or CPU)"""
    if torch.cuda.is_available():
        print("检测到 NVIDIA CUDA GPU，将使用 cuda 设备。")
        return "cuda"
    # 注意：torch.backends.mps.is_available() 在较新的PyTorch版本中用于检测M1/M2/M3芯片
    elif torch.backends.mps.is_available():
        print("检测到 Apple Silicon GPU (M1/M2/M3)，将使用 mps 设备。")
        return "mps"
    else:
        print("未检测到可用GPU，将使用 CPU。")
        return "cpu"

# --- 3. 定义所有模型名称和配置 ---
DEVICE = get_device()
WHISPER_MODEL = "small"
SUMMARIZER_MODEL = "microsoft/Phi-3-mini-4k-instruct"
SIMILARITY_MODEL = 'all-MiniLM-L6-v2'
# Docker镜像内的缓存目录，如果本地运行，则会下载到用户主目录的.cache下
CACHE_DIR = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))


def load_all_models():
    """
    在服务启动时调用，预先加载所有模型到内存/显存。
    这个函数现在是完整且带有详细错误处理的。
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

        # --- 加载 Qwen2-7B-Instruct 模型和分词器 (已补全) ---
        print(f"正在加载 Summarizer model: {SUMMARIZER_MODEL}...")
        
        # 根据不同设备，使用不同的加载策略
        if DEVICE == "cuda":
            # NVIDIA GPU: 使用4-bit量化以节省显存
            qwen_model = AutoModelForCausalLM.from_pretrained(
                SUMMARIZER_MODEL,
                torch_dtype="auto",
                device_map="auto",
                load_in_4bit=True,
                cache_dir=CACHE_DIR
            )
        elif DEVICE == "mps":
            # Apple Silicon GPU: 不支持4-bit量化，使用bfloat16精度加载
            qwen_model = AutoModelForCausalLM.from_pretrained(
                SUMMARIZER_MODEL,
                torch_dtype=torch.bfloat16, 
                device_map="auto",
                cache_dir=CACHE_DIR
            )
            # 将模型显式移动到mps设备
            qwen_model.to(DEVICE)
        else:
            # CPU: 使用默认精度加载
            qwen_model = AutoModelForCausalLM.from_pretrained(
                SUMMARIZER_MODEL, 
                cache_dir=CACHE_DIR
            ).to(DEVICE)

        # 加载对应的分词器
        qwen_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL, cache_dir=CACHE_DIR)
        
        # 将模型和分词器都存入注册表
        MODEL_REGISTRY['summarizer_model'] = qwen_model
        MODEL_REGISTRY['summarizer_tokenizer'] = qwen_tokenizer
        print("✅ Summarizer model (Qwen2) 加载完毕。")
        
        print("--- 所有模型均已成功加载并准备就绪！ ---")

    except Exception as e:
        print(f"❌ 模型加载过程中发生严重错误: {e}")
        # 在实际生产中，这里可能需要让整个应用启动失败
        raise e