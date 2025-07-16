import os
import sys
import whisper

def transcribe_file(model, audio_path):
    print(f"\n▶ 识别文件：{audio_path}")
    # 执行语音识别
    result = model.transcribe(audio_path, fp16=False)

    # 输出转写结果的其他顶层信息（排除 segments）
    print("—— 转写结果其他信息 ——")
    for key, val in result.items():
        if key != "segments":
            print(f"{key}: {val}")

    # 输出识别出的全文文本
    print("—— 转写文本 ——")
    print(result["text"])

    # 输出详细的时间片段信息
    print("—— 时间片段详情 ——")
    for seg in result["segments"]:
        print("-" * 40)
        for k, v in seg.items():
            print(f"{k}: {v}")

def main(filenames=None):
    # 1. 定位脚本目录和 audio 子目录
    base_dir  = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(base_dir, "audio")

    # 2. 加载模型
    model = whisper.load_model("small")

    # 3. 获取需要处理的音频文件列表
    if filenames:
        files = filenames
    else:
        files = [
            f for f in os.listdir(audio_dir)
            if os.path.splitext(f)[1].lower() in (".wav", ".mp3", ".ogg", ".webm")
        ]

    if not files:
        print("❌ 未找到音频文件，请确认 demo/audio/ 目录下存在支持格式文件。")
        return

    # 4. 对每个音频文件执行识别
    for fname in files:
        audio_path = os.path.join(audio_dir, fname)
        transcribe_file(model, audio_path)

if __name__ == "__main__":
    # 支持：
    #   python asr_demo.py             → 识别 demo/audio/ 下所有文件
    #   python asr_demo.py foo.wav     → 仅识别指定文件
    args = sys.argv[1:]
    main(filenames=args)