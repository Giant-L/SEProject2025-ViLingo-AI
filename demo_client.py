import requests
import time
import os

# --- 配置 ---
BASE_URL = "http://localhost:8000/api/v1"
ANALYZE_ENDPOINT = f"{BASE_URL}/analyze"
RESULTS_ENDPOINT = f"{BASE_URL}/results"

# --- 准备测试文件 (已修改) ---
# 将视频路径改为字幕文件路径
TEST_SRT_PATH = "test_data/test_subtitle.srt"
TEST_AUDIO_PATH = "test_data/test_audio.wav"

def run_demo():
    """运行一个完整的分析流程并打印结果。"""
    
    # 检查测试文件是否存在 (已修改)
    if not (os.path.exists(TEST_SRT_PATH) and os.path.exists(TEST_AUDIO_PATH)):
        print(f"错误：请确保测试文件存在于 '{TEST_SRT_PATH}' 和 '{TEST_AUDIO_PATH}'")
        return

    print("Step 1: 提交分析任务...")
    
    # 修改文件打开和上传的逻辑
    # .srt是文本文件，用 'r' 模式打开更标准
    with open(TEST_SRT_PATH, 'r', encoding='utf-8') as srt_f, open(TEST_AUDIO_PATH, 'rb') as audio_f:
        # 修改上传的字段名和文件
        files = {
            'srt_file': (os.path.basename(TEST_SRT_PATH), srt_f, 'text/plain'),
            'user_audio': (os.path.basename(TEST_AUDIO_PATH), audio_f, 'audio/wav'),
        }
        try:
            response = requests.post(ANALYZE_ENDPOINT, files=files, timeout=10)
            response.raise_for_status() # 如果状态码不是2xx，则抛出异常
        except requests.RequestException as e:
            print(f"提交任务失败: {e}")
            return

    job_data = response.json()
    job_id = job_data.get('job_id')
    print(f"任务提交成功！ Job ID: {job_id}\n")

    print("Step 2: 开始轮询任务结果 (每5秒一次)...")
    while True:
        try:
            response = requests.get(f"{RESULTS_ENDPOINT}/{job_id}", timeout=10)
            response.raise_for_status()
            status_data = response.json()
            
            status = status_data.get('status')
            print(f"  -> 当前状态: {status}")
            
            if status == 'completed':
                print("\n--- 分析完成！---\n")
                # 修改和增加打印的字段，以匹配新的API响应
                result = status_data.get('result', {})
                print(f"评分: {result.get('score')}")
                print(f"字幕摘要: {result.get('video_summary')}") # 将"视频"改为"字幕"更准确
                print(f"用户复述: {result.get('user_transcript')}")
                # 打印我们新增的 srt 原文字段
                print(f"字幕原文: {result.get('original_srt_text')}")
                break
            elif status == 'failed':
                print(f"\n--- 分析失败 ---")
                print(f"错误信息: {status_data.get('error')}")
                break
            elif status == 'processing':
                print(f"     详情: {status_data.get('stage')}")

        except requests.RequestException as e:
            print(f"查询结果失败: {e}")
            break
            
        time.sleep(5)

if __name__ == "__main__":
    # 温馨提示：在运行前，请确保 test_data 文件夹里有一个名为 test_subtitle.srt 的文件
    if not os.path.exists("test_data"):
        os.makedirs("test_data")
        print("提示：已创建 'test_data' 文件夹。请在其中放入 'test_subtitle.srt' 和 'test_audio.wav' 文件后再运行。")
    else:
        run_demo()