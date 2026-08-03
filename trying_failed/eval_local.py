import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config.settings import PROJECT_ROOT

tokenizer_path = PROJECT_ROOT / "models" / "deepseek-coder-1.3b-base"
model_path = PROJECT_ROOT / "sft_model"

print("[INFO] 加载模型到 CPU（本地 Mac）...")
tokenizer = AutoTokenizer.from_pretrained(
    str(tokenizer_path),
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    str(model_path),
    torch_dtype=torch.float32,        # Mac 用 float32
    device_map="cpu",                 # 显式指定 CPU
    trust_remote_code=True,
)
model.eval()

def clean_generated_text(text: str) -> str:
    """清理 BPE 特殊字符"""
    text = text.replace('Ċ', '\n')
    text = text.replace('Ġ', ' ')
    return text

def local_eval(prompt: str, max_new_tokens: int = 100):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_generated_text(raw)

# 测试 SFT 格式
test_cases = [
    {
        "instruction": "写一个计算器类 Calculator，包含加、减、乘、除四个方法",
        "expected": ["add", "subtract", "multiply", "divide"]
    },
    {
        "instruction": "实现一个函数判断一个数是否为素数",
        "expected": ["def is_prime", "for", "return"]
    },
    {
        "instruction": "用 pandas 读取 CSV 文件并显示前 5 行",
         "expected": ["pd.read_csv", "head()"]
    },
]

print("开始评估模型的代码能力...\n")
for case in test_cases:
    prompt = f"### Instruction:\n{case['instruction']}\n\n### Response:\n"
    local_eval(prompt, max_new_tokens=200)


