import torch
import math
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "sft_model"
TOKENIZER_PATH = PROJECT_ROOT / "models" / "deepseek-coder-1.3b-base"
DATA_PATH = PROJECT_ROOT / "data" / "training_samples.jsonl"


def format_prompt(instruction: str) -> str:
    return (
        "### Instruction:\n"
        f"{instruction.strip()}\n\n"
        "### Response:\n"
    )


def compute_perplexity_local():
    print("[INFO] 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_PATH),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print("[INFO] 加载数据...")
    dataset = load_dataset("json", data_files=str(DATA_PATH), split="train")
    dataset = dataset.filter(lambda x: x.get("type") == "code_comment")

    # 只取前 30 条做评估
    dataset = dataset.select(range(min(30, len(dataset))))
    print(f"[INFO] 评估样本数: {len(dataset)}")

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for idx, example in enumerate(dataset):
            instruction = example.get("instruction", "")
            code = example.get("code", "")

            # 构建完整文本
            prompt = format_prompt(instruction)
            full_text = prompt + code

            # 编码
            inputs = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(model.device)

            # 关键：计算 response 部分的起始位置
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_len = len(prompt_tokens)

            # 创建 labels：将 prompt 部分的 label 设为 -100（忽略）
            labels = inputs["input_ids"].clone()
            labels[:, :prompt_len] = -100

            # 前向传播
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss.item()

            # 只统计 response 部分的 token 数
            response_len = inputs["input_ids"].shape[1] - prompt_len
            total_loss += loss * response_len
            total_tokens += response_len

            if (idx + 1) % 10 == 0:
                print(f"[INFO] 已处理 {idx + 1}/{len(dataset)} 条")

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    print(f"\nSFT 模型困惑度: {perplexity:.4f}")
    print(f"   平均 Loss: {avg_loss:.4f}")
    return perplexity


if __name__ == "__main__":
    compute_perplexity_local()