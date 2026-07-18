# evaluation.py (修正后的计算困惑度部分)

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from tqdm import tqdm
from pathlib import Path
from config.settings import PROJECT_ROOT


model_path = Path(PROJECT_ROOT) / "final_model"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# 确保 tokenizer 有 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. 加载数据集
dataset = load_from_disk('./data/tokenized')
# 取前 200 条作为验证集
valid_dataset = dataset.select(range(min(200, len(dataset))))

losses = []
# 设置较小的批次大小，避免显存溢出
batch_size = 4

with torch.no_grad():
    for i in tqdm(range(0, len(valid_dataset), batch_size)):
        batch = valid_dataset[i:i + batch_size]

        # 关键步骤：使用 tokenizer.pad 对批次数据进行填充
        padded_batch = tokenizer.pad(
            {"input_ids": batch['input_ids']},
            padding=True,
            return_tensors="pt"
        )

        # 将填充后的数据移到模型设备
        inputs = padded_batch['input_ids'].to(model.device)

        # 计算损失
        outputs = model(inputs, labels=inputs)
        losses.append(outputs.loss.item())

# 计算并打印困惑度
if losses:
    avg_loss = np.mean(losses)
    perplexity = np.exp(avg_loss)
    print(f"验证集平均损失: {avg_loss:.4f}")
    print(f"验证集困惑度 (Perplexity): {perplexity:.2f}")
else:
    print("未计算到任何损失值，请检查数据集或批次大小。")
