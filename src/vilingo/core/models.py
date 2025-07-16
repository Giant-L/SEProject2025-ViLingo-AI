import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import whisper

# 使用一个字典作为简单的模型注册表，在应用生命周期中存储加载好的模型
MODEL_REGISTRY = {}

# 模型名称和设备配置
# 未来可以从配置文件或环境变量读取
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_MODEL = "small"
SUMMARIZER_MODEL = "Qwen/Qwen2-7B-Instruct"
SIMILARITY_MODEL = 'all-MiniLM-L6-v2'
CACHE_DIR = "/root/.cache" # Docker镜像内的缓存目录

def load_all_models():
    """
    在服务启动时调用，预先加载所有模型到内存/显存。
    """
    print(f"正在使用设备: {DEVICE}")

    # 1. 加载 Whisper Model
    print(f"正在加载 Whisper model: {WHISPER_MODEL}...")
    whisper_model = whisper.load_model(WHISPER_MODEL, device=DEVICE, download_root=f"{CACHE_DIR}/whisper")
    MODEL_REGISTRY['whisper'] = whisper_model
    print("Whisper model 加载完毕。")

    # 2. 加载 Sentence Transformer Model
    print(f"正在加载 Sentence Transformer model: {SIMILARITY_MODEL}...")
    st_model = SentenceTransformer(SIMILARITY_MODEL, device=DEVICE, cache_folder=CACHE_DIR)
    MODEL_REGISTRY['sentence_transformer'] = st_model
    print("Sentence Transformer model 加载完毕。")

    # 3. 加载 Qwen2-7B-Instruct Model (使用4-bit量化)
    print(f"正在加载 Summarizer model: {SUMMARIZER_MODEL} (4-bit)...")
    if DEVICE == "cuda":
        qwen_model = AutoModelForCausalLM.from_pretrained(
            SUMMARIZER_MODEL,
            torch_dtype="auto",
            device_map="auto",
            load_in_4bit=True,
            cache_dir=CACHE_DIR
        )
    else:
        # CPU模式下不建议使用4-bit量化
        qwen_model = AutoModelForCausalLM.from_pretrained(SUMMARIZER_MODEL, cache_dir=CACHE_DIR).to(DEVICE)

    qwen_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL, cache_dir=CACHE_DIR)
    
    MODEL_REGISTRY['summarizer_model'] = qwen_model
    MODEL_REGISTRY['summarizer_tokenizer'] = qwen_tokenizer
    print("Summarizer model 加载完毕。")