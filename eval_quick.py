import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from config.settings import PROJECT_ROOT

model_path = Path(PROJECT_ROOT) / "final_model"

tokenizer = AutoTokenizer.from_pretrained(
    str(model_path),
    trust_remote_code=True,
    use_fast=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    str(model_path),
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)


def clean_generated_text(text: str) -> str:
    """清理 DeepSeek-Coder BPE 特殊字符"""

    # 1. 替换 BPE 特殊字符
    text = text.replace('Ċ', '\n')  # Ċ → 换行
    text = text.replace('Ġ', ' ')  # Ġ → 空格

    # 2. 修复 BPE 分词导致的粘连（例如 "deffibonacci" → "def fibonacci"）
    # 注意：这些是常见的 BPE 分割模式，DeepSeek-Coder 会把 "def" 和 "fibonacci" 分开
    # 但有时候会粘在一起，需要手动修复
    text = re.sub(r'def([A-Za-z])', r'def \1', text)
    text = re.sub(r'class([A-Z])', r'class \1', text)
    text = re.sub(r'return([A-Za-z])', r'return \1', text)
    text = re.sub(r'import([A-Za-z])', r'import \1', text)
    text = re.sub(r'print([A-Za-z])', r'print \1', text)

    # 3. 清理多余空格（多个空格 → 单个空格，但保留缩进）
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # 计算前导空格数（缩进）
        leading_spaces = len(line) - len(line.lstrip())
        # 清理中间的多余空格，但保留前导缩进
        content = line.lstrip()
        content = re.sub(r' +', ' ', content)  # 多个空格 → 单个空格
        cleaned_lines.append(' ' * leading_spaces + content)

    text = '\n'.join(cleaned_lines)

    # 4. 修复可能的空行过多
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 5. 修复括号粘连
    text = text.replace('(', ' (')
    text = text.replace('( ', '(')  # 修复过度修复
    text = text.replace(' )', ')')

    return text.strip()


def quick_eval(prompt: str, max_new_tokens: int = 200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned = clean_generated_text(raw)

    print(f"\n{'=' * 60}")
    print(f"Prompt: {prompt}")
    print(f"{'=' * 60}")
    print(cleaned)
    print(f"{'=' * 60}")

    return cleaned


# 测试样本
test_prompts = [
    "def fibonacci(n):",
    "import pandas as pd\n\n# 读取 CSV 文件",
    "class Calculator:",
]

print("开始评估模型的代码能力...\n")
for prompt in test_prompts:
    quick_eval(prompt, max_new_tokens=200)