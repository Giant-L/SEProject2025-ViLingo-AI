# src/vilingo/core/pipeline.py (集成语言检测的最终版)

import torch
import os
import shutil
from .models import MODEL_REGISTRY

# --- 【修改】概括任务的指令模板，改为英文，以支持英文对英文的对比 ---
SUMMARIZATION_PROMPT_TEMPLATE = """You are an expert content analyst, skilled at distilling the core points and logical framework from complex texts. Please summarize the following text into a coherent and fluid summary of no more than 200 words.

The text is as follows:
---
{text}
---

Your summary:"""

def _run_summarization(text: str) -> str:
    """内部函数：调用LLM执行概括任务"""
    model = MODEL_REGISTRY['summarizer_model']
    tokenizer = MODEL_REGISTRY['summarizer_tokenizer']
    device = model.device

    # 使用英文Prompt
    prompt = SUMMARIZATION_PROMPT_TEMPLATE.format(text=text)
    
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    summary = response[len(prompt):].strip()
    return summary

def _parse_srt_to_text(srt_content: str) -> str:
    """
    从SRT文件内容中提取所有对话文本，并合并成一个字符串。
    """
    lines = srt_content.strip().split('\n')
    text_lines = []
    for line in lines:
        if line.strip() and not line.strip().isdigit() and '-->' not in line:
            text_lines.append(line.strip())
    return ' '.join(text_lines)


def execute_analysis_pipeline(job_id: str, srt_path: str, user_audio_path: str, results_db: dict):
    """
    这是在后台任务中执行的核心函数。
    集成了用户录音的语言检测功能。
    """
    temp_dir = os.path.dirname(srt_path)
    
    try:
        # --- 初始化模型 ---
        whisper_model = MODEL_REGISTRY['whisper']
        st_model = MODEL_REGISTRY['sentence_transformer']
        use_fp16 = torch.cuda.is_available()

        # --- 阶段一: 字幕概括 ---
        results_db[job_id] = {"status": "processing", "stage": "Parsing and summarizing subtitle..."}
        with open(srt_path, 'r', encoding='utf-8') as f:
            srt_content = f.read()
        video_transcript = _parse_srt_to_text(srt_content)
        if not video_transcript:
            raise ValueError("字幕文件中未能解析出任何文本内容。")
        # 使用新的英文Prompt生成英文摘要
        video_summary = _run_summarization(video_transcript)

        # --- 阶段二: 用户录音处理与语言检测 ---
        results_db[job_id] = {"status": "processing", "stage": "Transcribing and detecting language..."}
        
        # --- 【修改】获取完整的转录结果，而不仅仅是文本 ---
        user_transcription_result = whisper_model.transcribe(user_audio_path, fp16=use_fp16)
        
        user_transcript = user_transcription_result["text"]
        detected_language = user_transcription_result["language"]

        print(f"任务 {job_id}: 检测到用户语言为 '{detected_language}'")

        # --- 【新增】语言检测关卡 ---
        if detected_language != 'en':
            # 如果语言不是英文，构造一个特定的低分结果并立即完成任务
            error_result = {
                "score": 0, # 直接给0分
                "video_summary": video_summary,
                "user_transcript": user_transcript, # 仍然返回识别出的非英文内容
                "original_srt_text": video_transcript,
                "feedback": "未能检测到有效的英文复述，请使用英文进行表达。 (Invalid language detected. Please use English for your retelling.)"
            }
            results_db[job_id] = {"status": "completed", "result": error_result}
            print(f"任务 {job_id} 因语言不符而提前终止。")
            return # 使用 return 提前结束函数，不再执行后续步骤
        
        if not user_transcript:
            raise ValueError("用户录音中未能识别出任何语音内容。")

        # --- 阶段三: 内容比对与评分 (只有语言为英文时才会执行) ---
        results_db[job_id] = {"status": "processing", "stage": "Comparing texts and scoring..."}
        from sentence_transformers.util import cos_sim
        embedding_summary = st_model.encode(video_summary, convert_to_tensor=True)
        embedding_user = st_model.encode(user_transcript, convert_to_tensor=True)
        
        similarity = cos_sim(embedding_summary, embedding_user)
        score = round(float(similarity[0][0]) * 100, 2)
        
        # 为成功的结果也增加 feedback 字段，保持格式统一
        final_result = {
            "score": score,
            "video_summary": video_summary,
            "user_transcript": user_transcript,
            "original_srt_text": video_transcript,
            "feedback": "评分成功！ (Scoring successful!)"
        }
        
        results_db[job_id] = {"status": "completed", "result": final_result}
        print(f"任务 {job_id} 成功完成。")

    except Exception as e:
        print(f"任务 {job_id} 失败: {e}")
        results_db[job_id] = {"status": "failed", "error": str(e)}
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"已清理临时目录: {temp_dir}")