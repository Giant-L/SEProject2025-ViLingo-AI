# demo/asr_demo.py
import os
import sys
import whisper

def transcribe_file(model, audio_path):
    print(f"\n▶ 识别文件：{audio_path}")
    result = model.transcribe(audio_path, fp16=False)
    print("—— 转写文本 ——")
    print(result["text"])
    print("—— 时间片段 ——")
    for seg in result["segments"]:
        print(f"  [{seg['start']:.2f}s→{seg['end']:.2f}s] {seg['text']}")

def main(filenames=None):
    # 1. 定位脚本目录和 audio 子目录
    base_dir  = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(base_dir, "audio")

    # 2. 加载模型
    model = whisper.load_model("small")

    # 3. 如果指定了文件名，就只识别它；否则遍历整个 audio 目录
    if filenames:
        files = filenames
    else:
        # 只挑 .wav/.mp3/.ogg/.webm 文件
        files = [f for f in os.listdir(audio_dir)
                 if os.path.splitext(f)[1].lower() in (".wav", ".mp3", ".ogg", ".webm")]

    if not files:
        print("❌ 没有找到任何音频文件，请确认 demo/audio/ 目录下有 .wav/.mp3/.ogg/.webm 文件。")
        return

    # 4. 对每个文件执行转写
    for fname in files:
        audio_path = os.path.join(audio_dir, fname)
        transcribe_file(model, audio_path)

if __name__ == "__main__":
    # 支持：python asr_demo.py            → 识别 demo/audio/ 下所有文件
    #       python asr_demo.py foo.wav     → 只识别 foo.wav
    args = sys.argv[1:]
    main(filenames=args)