
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from config.settings import PROJECT_ROOT

model_path = Path(PROJECT_ROOT) / "final_model"
tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    str(model_path),
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 测试样本
test_prompts = [
    "def fibonacci(n):",
    "import pandas as pd\n\n# 读取 CSV 文件",
    "class Calculator:",
]

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n--- Prompt: {prompt} ---")
    print(generated)
