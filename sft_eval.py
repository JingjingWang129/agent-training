import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
model_path = Path(PROJECT_ROOT) / "sft_model"

tokenizer = AutoTokenizer.from_pretrained(
    str(model_path),
    trust_remote_code=True,
    use_fast=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    str(model_path),
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)


def fix_compact_code(code: str) -> str:
    """修复粘连的 Python 代码，添加必要的空格和换行"""

    # 添加空格分隔
    # 1. 关键字粘连: defis_prime → def is_prime
    keywords = ['def', 'class', 'if', 'elif', 'else', 'for', 'while',
                'try', 'except', 'finally', 'with', 'return', 'yield',
                'raise', 'import', 'from', 'print', 'assert', 'pass',
                'break', 'continue', 'global', 'nonlocal', 'del']
    for kw in keywords:
        code = re.sub(rf'({kw})([A-Za-z_])', r'\1 \2', code)

    # 2. 类型注解: n:int → n: int, )->bool → ) -> bool
    code = re.sub(r':([A-Za-z_])', r': \1', code)
    code = re.sub(r'\)\(', r') (', code)  # )( → ) (
    code = re.sub(r'\)->', r') -> ', code)  # )-> → ) ->
    code = re.sub(r'->([A-Za-z_])', r'-> \1', code)

    # 3. 操作符间距: n<=1 → n <= 1, n%2 → n % 2
    ops = ['<=', '>=', '==', '!=', '<<', '>>', '**', '//']
    for op in ops:
        code = re.sub(rf'([a-zA-Z0-9_]){re.escape(op)}([a-zA-Z0-9_])', rf'\1 {op} \2', code)

    single_ops = ['=', '+', '-', '*', '/', '%', '<', '>', '&', '|', '^']
    for op in single_ops:
        code = re.sub(rf'([a-zA-Z0-9_]){re.escape(op)}([a-zA-Z0-9_])', rf'\1 {op} \2', code)

    # 4. 逗号分隔: range(3,int(...)) → range(3, int(...))
    code = re.sub(r',([A-Za-z_])', r', \1', code)
    code = re.sub(r',([0-9])', r', \1', code)

    # 5. 括号与关键字: if(n → if (n, for i in range → for i in range (已经是好的)
    code = re.sub(r'if\(', r'if (', code)
    code = re.sub(r'elif\(', r'elif (', code)
    code = re.sub(r'while\(', r'while (', code)

    # 6. 参数默认值: a=1 → a = 1
    # 只在函数定义和调用中处理
    code = re.sub(r'(self\.)?([a-zA-Z_][a-zA-Z0-9_]*)=([a-zA-Z0-9_"\'])', r'\1\2 = \3', code)

    # 在关键字前添加换行（如果还没有换行）
    lines = code.split('\n')
    if len(lines) == 1:
        # 只有一行，在关键字前插入换行
        for kw in ['def ', 'class ', 'if ', 'elif ', 'else:', 'for ', 'while ',
                   'try:', 'except ', 'finally:', 'with ', '@']:
            code = re.sub(rf'({kw})', r'\n\1', code)

    # 处理缩进
    lines = code.split('\n')
    result = []
    indent_level = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 检测是否需要增加缩进
        if line.startswith(('class ', 'def ', 'if ', 'elif ', 'else:', 'for ',
                            'while ', 'try:', 'except ', 'finally:', 'with ', '@')):
            # 如果上一行不是空行，添加空行分隔
            if result and result[-1].strip():
                result.append('')
            result.append('    ' * indent_level + line)
            if line.endswith(':') or line.startswith(('class ', 'def ')):
                indent_level += 1
        else:
            result.append('    ' * indent_level + line)

    return '\n'.join(result)


def clean_and_format_response(raw: str) -> str:
    """清理并格式化模型输出"""
    # 移除 Instruction/Response 标记
    text = re.sub(r'###\s*Instruction:.*?###\s*Response:', '', raw, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text)
    text = text.strip()

    # 修复代码
    return fix_compact_code(text)


def quick_eval(prompt: str, max_new_tokens: int = 300):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    formatted = clean_and_format_response(raw)

    print(f"\n{'=' * 60}")
    print(f" Prompt: {prompt}")
    print(f"{'=' * 60}")
    print(formatted)
    print(f"{'=' * 60}")

    return formatted


# 测试
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