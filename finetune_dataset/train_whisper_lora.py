import os
import torch
from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# ====================================================================================
# 💡 核心修改：动态路径管理
# 无论从哪里运行此脚本，都能确保路径正确
# ====================================================================================
# 1. 获取此脚本文件所在的绝对路径
script_path = os.path.abspath(__file__)
# 2. 从脚本路径推断出项目根目录 (finetune_dataset 目录的上级)
project_root = os.path.dirname(os.path.dirname(script_path))
print(f"✅ 项目根目录已自动设置为: {project_root}")

# 3. 基于项目根目录定义所有其他路径
data_file = os.path.join(project_root, "finetune_dataset", "train.jsonl")
cache_dir = os.path.join(project_root, ".cache") # 建议将cache放在根目录
output_dir = os.path.join(project_root, "finetune_dataset", "output")
save_dir = os.path.join(project_root, "finetune_dataset", "finetuned-whisper-lora")
# ====================================================================================


# Step 1: 加载 Whisper 模型和处理器
model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Step 2: 加载数据集 (使用我们定义的动态路径)
print(f"⏳ 正在从 '{data_file}' 加载数据集...")
dataset = load_dataset("json", data_files=data_file, split="train", cache_dir=cache_dir)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
print("✅ 数据集加载完成")

# Step 3: 添加 LoRA adapter
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Step 4: 数据预处理函数
def prepare_dataset(batch):
    # 注意：这里的 audio 路径是相对 'data_file' 的，datasets 库会自动处理
    # 我们不需要在这里修改它
    audio = batch["audio"]
    batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

# Step 5: 映射预处理
processed_dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names, num_proc=1)

# Step 6: 数据整理器 (这部分代码是正确的，保持不变)
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

# Step 7: 实例化数据整理器
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


# Step 8: 自定义 Trainer (保持不变)
class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if 'input_ids' in inputs:
            del inputs['input_ids']
        return super().compute_loss(model, inputs, return_outputs)

# Step 9: 定义训练参数 (使用我们定义的动态路径)
use_fp16 = torch.cuda.is_available()

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=5,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    evaluation_strategy="no",
    fp16=use_fp16,
    report_to="none",
    push_to_hub=False,
)

# Step 10: 创建我们自定义的 Trainer 实例
trainer = CustomSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)

print("\n--- 开始训练 ---")
trainer.train()

# Step 11: 保存模型 (使用我们定义的动态路径)
print(f"\n✅ 训练完成! 模型将保存到: {save_dir}")
model.save_pretrained(save_dir)
processor.save_pretrained(save_dir)
print("🎉 模型和处理器已成功保存!")