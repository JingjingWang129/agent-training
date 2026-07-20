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

# 诊断脚本
test_text = "def fibonacci(n):"
tokens = tokenizer.encode(test_text)
print(f"Tokens: {tokens}")
print(f"Token 对应的文本: {[tokenizer.decode([t]) for t in tokens]}")

# 测试 decode
decoded_default = tokenizer.decode(tokens, skip_special_tokens=True)
decoded_with_spaces = tokenizer.decode(tokens, skip_special_tokens=True, spaces_between_special_tokens=True)

print(f"默认 decode: {repr(decoded_default)}")
print(f"带 spaces 参数: {repr(decoded_with_spaces)}")
print(f"原始文本: {repr(test_text)}")