import requests
import uuid

BASE_URL = "http://localhost:8000/api/v1"

def test_invalid_path():
    """测试一个错误的URL路径，预期得到 404 Not Found"""
    print("--- 正在测试错误的API路径 ---")
    
    # 模拟前端最常见的错误：丢失 /api/v1 前缀
    invalid_url = "http://localhost:8000/analyze"
    print(f"尝试调用 (错误路径): POST {invalid_url}")

    try:
        # 准备假的上传文件
        files = {
            'summary_txt_file': ('summary.txt', 'some summary text', 'text/plain'),
            'user_audio': ('audio.mp3', b'some_audio_data', 'audio/mpeg'),
        }
        response = requests.post(invalid_url, files=files, timeout=5)
        
        print(f"服务器返回状态码: {response.status_code}")
        if response.status_code == 404:
            print("✅ 测试成功！服务器正确返回了 404 Not Found。")
        else:
            print(f"❌ 测试失败！预期状态码为 404，但收到了 {response.status_code}。")
            print("响应内容:", response.text)

    except requests.ConnectionError:
        print("❌ 连接错误：请确保您的Docker服务正在运行中。")
    except Exception as e:
        print(f"❌ 测试时发生未知错误: {e}")
    print("-" * 30)


def test_nonexistent_job_id():
    """测试查询一个不存在的 job_id，预期得到 404 Not Found"""
    print("\n--- 正在测试不存在的 Job ID ---")
    
    # 生成一个随机的、不存在的 UUID 作为 job_id
    fake_job_id = str(uuid.uuid4())
    url = f"{BASE_URL}/results/{fake_job_id}"
    print(f"尝试调用: GET {url}")

    try:
        response = requests.get(url, timeout=5)
        
        print(f"服务器返回状态码: {response.status_code}")
        if response.status_code == 404:
            print("✅ 测试成功！服务器正确返回了 404 Not Found。")
        else:
            print(f"❌ 测试失败！预期状态码为 404，但收到了 {response.status_code}。")
            print("响应内容:", response.text)

    except requests.ConnectionError:
        print("❌ 连接错误：请确保您的Docker服务正在运行中。")
    except Exception as e:
        print(f"❌ 测试时发生未知错误: {e}")
    print("-" * 30)

if __name__ == "__main__":
    # 在运行此脚本前，请确保您的Docker容器正在运行
    test_invalid_path()
    test_nonexistent_job_id()