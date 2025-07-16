import os
import json

wav_dir = "finetune_dataset/audio"
transcript_dir = "database/HKK/transcript"
output_path = "finetune_dataset/train.jsonl"

with open(output_path, "w", encoding="utf-8") as outfile:
    for filename in sorted(os.listdir(wav_dir)):
        if not filename.endswith(".wav"):
            continue
        base = os.path.splitext(filename)[0]
        wav_path = os.path.join("audio", filename)  # 使用相对路径
        txt_path = os.path.join(transcript_dir, base + ".txt")

        if not os.path.exists(txt_path):
            print(f"Skipping {filename} - no transcript")
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        example = {"audio": wav_path, "text": text}
        outfile.write(json.dumps(example, ensure_ascii=False) + "\n")