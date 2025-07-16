from .models import MODEL_REGISTRY
import torch

# 概括任务的指令模板
SUMMARIZATION_PROMPT_TEMPLATE = """你是一位内容分析专家，擅长从复杂的文本中精准地提炼出核心要点和逻辑框架。请将以下文本概括成一段不超过200字的、连贯流畅的摘要。

文本内容如下：
---
{text}
---

请输出你的摘要："""

def run_summarization(text: str) -> str:
    """调用LLM执行概括任务"""
    model = MODEL_REGISTRY['summarizer_model']
    tokenizer = MODEL_REGISTRY['summarizer_tokenizer']
    device = model.device

    prompt = SUMMARIZATION_PROMPT_TEMPLATE.format(text=text)
    
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512, # 限制最大生成长度
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # 清理，只返回prompt之后生成的内容
    summary = response[len(prompt):].strip()
    return summary

def run_analysis_pipeline(video_path: str, user_audio_path: str) -> dict:
    """
    执行完整的AI分析流水线
    返回一个包含分数、摘要和转录文本的字典
    """
    whisper_model = MODEL_REGISTRY['whisper']
    st_model = MODEL_REGISTRY['sentence_transformer']
    use_fp16 = torch.cuda.is_available()

    # 阶段一: 视频概括
    print("Pipeline: 正在转录视频...")
    video_transcript = whisper_model.transcribe(video_path, fp16=use_fp16)["text"]
    if not video_transcript:
        raise ValueError("视频中未能识别出任何语音内容。")

    print("Pipeline: 正在概括视频内容...")
    video_summary = run_summarization(video_transcript)

    # 阶段二: 用户录音处理
    print("Pipeline: 正在转录用户录音...")
    user_transcript = whisper_model.transcribe(user_audio_path, fp16=use_fp16)["text"]
    if not user_transcript:
        raise ValueError("用户录音中未能识别出任何语音内容。")

    # 阶段三: 内容比对与评分
    print("Pipeline: 正在计算语义相似度...")
    from sentence_transformers.util import cos_sim
    embedding_summary = st_model.encode(video_summary, convert_to_tensor=True)
    embedding_user = st_model.encode(user_transcript, convert_to_tensor=True)
    
    similarity = cos_sim(embedding_summary, embedding_user)
    score = round(float(similarity[0][0]) * 100, 2)
    
    return {
        "score": score,
        "video_summary": video_summary,
        "user_transcript": user_transcript
    }