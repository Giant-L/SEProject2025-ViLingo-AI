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

# Step 1: 加载 Whisper 模型和处理器
model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Step 2: 加载数据集
dataset = load_dataset("json", data_files="finetune_dataset/train.jsonl", split="train", cache_dir="./.cache")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

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


# ✅✅✅ Step 8: 最终解决方案 -> 创建自定义 Trainer ✅✅✅
class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        重写 compute_loss 函数。
        这是 Trainer 将数据传递给模型前的最后一步。
        我们在这里拦截 'inputs' 字典并进行修正。
        """
        # 🕵️‍♂️ 最终诊断: 检查进入 compute_loss 的 inputs
        print("[DIAGNOSTIC] Inside custom compute_loss. Keys in 'inputs':", list(inputs.keys()))

        # 如果 'input_ids' 意外地出现在这里，就强制删除它
        if 'input_ids' in inputs:
            print("!!! WARNING: 'input_ids' found in inputs dict inside compute_loss. Forcibly removing it. !!!")
            del inputs['input_ids']

        # 调用原始的 compute_loss 函数来完成实际的计算
        return super().compute_loss(model, inputs, return_outputs)

# Step 9: 定义训练参数
use_fp16 = torch.cuda.is_available()

training_args = Seq2SeqTrainingArguments(
    output_dir="./finetune_dataset/output",
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
trainer = CustomSeq2SeqTrainer( # <-- 使用我们的自定义 Trainer
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)

print("\n--- Starting Training with Custom Trainer ---")
trainer.train()

# Step 11: 保存模型
print("\n--- Training Finished Successfully! Saving model... ---")
model.save_pretrained("./finetune_dataset/finetuned-whisper-lora")
processor.save_pretrained("./finetune_dataset/finetuned-whisper-lora")