import torch
import re
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from pathlib import Path
from config.settings import PROJECT_ROOT

model_path = Path(PROJECT_ROOT) / "new_sft_model"

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=str(model_path / "tokenizer.json"),
    bos_token="<｜begin▁of▁sentence｜>",
    eos_token="<｜end▁of▁sentence｜>",
    pad_token="<｜end▁of▁sentence｜>",
    clean_up_tokenization_spaces=False,
    model_max_length=16384,
)

model = AutoModelForCausalLM.from_pretrained(
    str(model_path),
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)


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

    print(f"\n{'=' * 60}")
    print(f"Prompt: {prompt}")
    print(f"{'=' * 60}")
    print(raw)
    print(f"{'=' * 60}")

    return raw


# 测试样本 -- python指令
# test_prompts = [
#     "def fibonacci(n):",
#     "import pandas as pd\n\n# 读取 CSV 文件",
#     "class Calculator:",
# ]

# print("开始评估模型的代码能力...\n")
# for prompt in test_prompts:
#     quick_eval(prompt, max_new_tokens=200)

# 测试样本 -- 自然语言指令
test_cases = [
    {
        "instruction": "Implement a function to determine whether a number is a prime number",
    },
    {
        "instruction": "Write a calculator class named Calculator, which includes four methods: addition, subtraction, multiplication, and division",
    },
    {
        "instruction": "Use pandas to read the CSV file and display the first 5 rows",
    },
]

print("开始评估 SFT 模型的代码能力...\n")
for case in test_cases:
    prompt = f"### Instruction:\n{case['instruction']}\n\n### Response:\n"
    quick_eval(prompt, max_new_tokens=300)