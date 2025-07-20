import requests
import time
import os

# --- 配置 ---
BASE_URL = "http://localhost:8000/api/v1"
ANALYZE_ENDPOINT = f"{BASE_URL}/analyze"
RESULTS_ENDPOINT = f"{BASE_URL}/results"

# --- 准备测试文件 (已修改为 .txt) ---
# 将 srt 路径改为 summary.txt 的路径
TEST_SUMMARY_PATH = "test_data/summary.txt" 
TEST_AUDIO_PATH = "test_data/test_audio.wav"

def run_demo():
    """运行一个完整的多维度分析流程并打印结果。"""
    
    # 检查测试文件是否存在 (已修改)
    if not (os.path.exists(TEST_SUMMARY_PATH) and os.path.exists(TEST_AUDIO_PATH)):
        print(f"错误：请确保测试文件存在于 '{TEST_SUMMARY_PATH}' 和 '{TEST_AUDIO_PATH}'")
        return

    print("Step 1: 提交分析任务...")
    
    # 修改文件打开和上传的逻辑
    with open(TEST_SUMMARY_PATH, 'r', encoding='utf-8') as summary_f, open(TEST_AUDIO_PATH, 'rb') as audio_f:
        # 修改上传的字段名和文件，以匹配最新的 main.py
        files = {
            'summary_txt_file': (os.path.basename(TEST_SUMMARY_PATH), summary_f, 'text/plain'),
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
                # 更新打印逻辑，以展示多维度的评分结果
                result = status_data.get('result', {})
                print(f"✨ 综合评分: {result.get('overall_score')}\n")
                
                print("--- 分数详情 ---")
                breakdown = result.get('score_breakdown', {})
                print(f"  - 内容相似度: {breakdown.get('content_similarity')}")
                print(f"  - 口语流畅度: {breakdown.get('fluency')}")
                print(f"  - 语法准确度: {breakdown.get('grammar_accuracy')}")
                print("-----------------\n")

                print(f"原始摘要: {result.get('original_summary')}")
                print(f"用户复述: {result.get('user_transcript')}")
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
    # 更新温馨提示
    if not os.path.exists("test_data"):
        os.makedirs("test_data")
        print("提示：已创建 'test_data' 文件夹。请在其中放入 'summary.txt' 和 'test_audio.wav' 文件后再运行。")
    else:
        run_demo()