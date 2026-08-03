import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
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


def quick_eval(prompt: str, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
        no_repeat_ngram_size=6,
    )

    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\n{'=' * 60}")
    print(f"Prompt: {prompt}")
    print(f"{'=' * 60}")
    print(raw)
    print(f"{'=' * 60}")

    return raw


# 测试样本 -- 自然语言指令
# test_cases = [
#     {
#         "instruction": "Implement a function to determine whether a number is a prime number",
#     },
#     {
#         "instruction": "Write a calculator class named Calculator, which includes four methods: addition, subtraction, multiplication, and division",
#     },
#     {
#         "instruction": "Use pandas to read the CSV file and display the first 5 rows",
#     },
# ]

# 测试样本 -- 基础逻辑
# test_cases = [
#     {
#         "instruction": "Write a function to determine whether a string is a palindrome",
#     },
#     {
#         "instruction": "Implement a function to calculate the greatest common divisor (GCD) of two numbers",
#     },
#     {
#         "instruction": "Write a function to remove duplicates from a list of numbers and return the new list after removing duplicates",
#     },
# {
#         "instruction":"Implement a function that capitalizes the first letter of each word in the input string",
#     },
#     {
#         "instruction":"Write a function to count the occurrence of each character in a string and return a dictionary",
#     },
# ]

# 测试样本 -- 经典算法
# test_cases = [
#     {
#         "instruction": "Implement a binary search algorithm to find the target value in an ordered array and return its index",
#     },
#     {
#         "instruction": "Write a function to implement the quick sort algorithm for sorting a list in ascending order",
#     },
#     {
#         "instruction": "Implement a Fibonacci sequence generator that returns the nth Fibonacci number",
#     },
# {
#         "instruction":"Implement a function that merges two sorted lists and returns a new sorted list",
#     },
# ]

# # 测试样本 -- 数据结构
# test_cases = [
#     {
#         "instruction": "Define a Stack class, implementing methods for pushing, popping, peeking at the top element, and checking whether it is empty",
#     },
#     {
#         "instruction": "Implement a function to find the element that appears most frequently in a list",
#     },
#     {
#         "instruction": "Write a function that swaps the keys and values of a dictionary to generate a new dictionary",
#     },
# {
#         "instruction":"Implement a simple LinkedList class, including methods for adding nodes and traversing and printing the list",
#     },
#     {
#         "instruction":"Implement a function that converts two lists into a dictionary, where one list serves as the keys and the other as the values",
#     },
# ]

# # 测试样本 -- 多步骤推理
test_cases = [
    {
        "instruction": "Write a function that takes a list containing multiple numbers and returns a new list consisting of the squares of all the even numbers in the original list",
    },
    {
        "instruction": "Implement a function that reads a text file, counts the occurrence of each word in it, and returns the top 5 words with the highest occurrence",
    },
    {
        "instruction": "Write a simple decorator that calculates the execution time of a function and prints out the elapsed time",
    },
    {
        "instruction": "Implement a function that flattens a nested list (of arbitrary depth) into a one-dimensional list",
    },
    {
        "instruction": "Write a function that takes a date string (format: YYYY-MM-DD) and calculates and returns the day of the year it falls on",
    },
]

print("开始评估 SFT 模型的代码能力...\n")
for case in test_cases:
    prompt = f"### Instruction:\n{case['instruction']}\n\n### Response:\n"
    quick_eval(prompt)