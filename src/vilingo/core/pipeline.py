# src/vilingo/core/pipeline.py (集成新评分逻辑的最终版)

import torch
import os
import shutil
from .models import MODEL_REGISTRY
import language_tool_python
import math # 新增：导入数学库以使用开根号功能

# --- 新增：初始化语法检查工具 (在容器启动时加载一次) ---
lang_tool = language_tool_python.LanguageTool('en-US')

# --- 权重配置 ---
WEIGHT_CONTENT = 0.5
WEIGHT_FLUENCY = 0.2
WEIGHT_GRAMMAR = 0.3

# --- 新增：流畅度评分函数 ---
def _calculate_fluency_score(transcription_result: dict) -> float:
    """基于Whisper的词时间戳计算流畅度得分"""
    words = []
    for segment in transcription_result.get('segments', []):
        words.extend(segment.get('words', []))

    if not words:
        return 0.0

    total_duration = words[-1]['end'] - words[0]['start']
    word_count = len(words)
    
    # 1. 语速 (Words Per Minute)
    wpm = (word_count / total_duration) * 60 if total_duration > 0 else 0
    # 理想区间 120-180 WPM, 在此区间内得分高
    if 120 <= wpm <= 180:
        pace_score = 100.0
    elif wpm < 120:
        pace_score = max(0, (wpm / 120) * 100)
    else:
        pace_score = max(0, (180 / wpm) * 100)

    # 2. 停顿 (Pauses) - 简单版：计算长停顿次数
    long_pauses = 0
    for i in range(word_count - 1):
        pause_duration = words[i+1]['start'] - words[i]['end']
        if pause_duration > 1.0: # 超过1秒算长停顿
            long_pauses += 1
    
    # 每分钟长停顿次数越多，得分越低
    pauses_per_minute = (long_pauses / total_duration) * 60 if total_duration > 0 else 0
    pause_score = max(0, 100 - (pauses_per_minute * 20)) # 每分钟5次长停顿扣完

    return (pace_score * 0.6) + (pause_score * 0.4)

# --- 新增：语法评分函数 ---
def _calculate_grammar_score(text: str) -> float:
    """使用 language-tool-python 检查语法错误并评分"""
    matches = lang_tool.check(text)
    error_count = len(matches)
    word_count = len(text.split())

    if word_count == 0:
        return 0.0

    # 计算每百词错误率
    errors_per_100_words = (error_count / word_count) * 100
    
    # 错误率越高，分数越低。每百词5个错误扣完。
    score = max(0, 100 - (errors_per_100_words * 20))
    return score

# --- 内容评分函数 ---
def _calculate_semantic_score(text1: str, text2: str) -> float:
    """计算两个文本的语义相似度得分"""
    st_model = MODEL_REGISTRY['sentence_transformer']
    from sentence_transformers.util import cos_sim
    
    embedding1 = st_model.encode(text1, convert_to_tensor=True)
    embedding2 = st_model.encode(text2, convert_to_tensor=True)
    
    similarity = cos_sim(embedding1, embedding2)
    score = round(float(similarity[0][0]) * 100, 2)
    return score


def execute_analysis_pipeline(job_id: str, summary_path: str, user_audio_path: str, results_db: dict):
    """
    执行完整的多维度分析流水线
    """
    temp_dir = os.path.dirname(summary_path)
    
    try:
        whisper_model = MODEL_REGISTRY['whisper']
        use_fp16 = torch.cuda.is_available()

        # ... (阶段一和阶段二的代码保持不变) ...
        results_db[job_id] = {"status": "processing", "stage": "Reading summary text..."}
        with open(summary_path, 'r', encoding='utf-8') as f:
            video_summary = f.read()
        if not video_summary:
            raise ValueError("摘要文本文件内容为空。")

        results_db[job_id] = {"status": "processing", "stage": "Transcribing and detecting language..."}
        user_transcription_result = whisper_model.transcribe(user_audio_path, fp16=use_fp16, word_timestamps=True)
        detected_language = user_transcription_result["language"]
        print(f"任务 {job_id}: 检测到用户语言为 '{detected_language}'")

        if detected_language != 'en':
            raise ValueError(f"语言错误：需要英文复述，但检测到 {detected_language}。")
        
        user_transcript = user_transcription_result["text"]
        if not user_transcript:
            raise ValueError("用户录音中未能识别出任何语音内容。")
            
        # --- 阶段三: 三个维度并行计算 ---
        results_db[job_id] = {"status": "processing", "stage": "Calculating scores..."}
        
        score_content = _calculate_semantic_score(video_summary, user_transcript)
        score_fluency = _calculate_fluency_score(user_transcription_result)
        score_grammar = _calculate_grammar_score(user_transcript)

        # --- 【核心修改】应用新的评分逻辑 ---
        # 1. 计算原始的加权总分 (0-100)
        raw_overall_score = (score_content * WEIGHT_CONTENT) + \
                            (score_fluency * WEIGHT_FLUENCY) + \
                            (score_grammar * WEIGHT_GRAMMAR)
        
        # 2. 应用变换：开根号再乘以10
        #    这会将分数曲线变得更平滑，低分区间的差异被放大，高分区间的差异被缩小
        transformed_score = math.sqrt(raw_overall_score) * 10
        
        # 3. 设置分数下限，避免出现过低的分数
        #    如果变换后的分数低于45，则强制拉高到45分
        #    我们用一个随机的小数来避免每次都是完全相同的整数
        import random
        if transformed_score < 45:
            final_score = 45.0 + random.uniform(0, 2) # 结果在 45.0 ~ 47.0 之间
        else:
            final_score = transformed_score
        
        # ------------------------------------
        
        # --- 组装最终结果 ---
        final_result = {
            "overall_score": round(final_score, 2), # 使用我们最终计算的分数
            "score_breakdown": {
                "content_similarity": round(score_content, 2),
                "fluency": round(score_fluency, 2),
                "grammar_accuracy": round(score_grammar, 2)
            },
            "original_summary": video_summary,
            "user_transcript": user_transcript,
        }
        
        results_db[job_id] = {"status": "completed", "result": final_result}
        print(f"任务 {job_id} 成功完成。原始加权分: {raw_overall_score:.2f}, 最终调整分: {final_result['overall_score']:.2f}")

    except Exception as e:
        print(f"任务 {job_id} 失败: {e}")
        results_db[job_id] = {"status": "failed", "error": str(e)}
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"已清理临时目录: {temp_dir}")