import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent
# MODEL_PATH = PROJECT_ROOT / "final_model"
# MODEL_PATH = PROJECT_ROOT / "sft_model_v2"
MODEL_PATH = PROJECT_ROOT / "models" / "deepseek-coder-1.3b-base"
DATA_PATH = PROJECT_ROOT / "data" / "training_samples.jsonl"

MAX_LENGTH = 2048
SAMPLE_SIZE = 5

IGNORE_INDEX = -100


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def build_text(example: Dict[str, str]) -> str:
    instruction = str(example.get("instruction") or "").strip()
    code = str(example.get("code") or "").strip()

    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
        "```python\n"
        f"{code}\n"
        "```"
    )


def build_prompt(example: Dict[str, str]) -> str:
    instruction = str(example.get("instruction") or "").strip()

    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
        "```python\n"
    )


def load_samples() -> List[Dict[str, str]]:
    samples = []

    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)

            if item.get("type") != "code_comment":
                continue

            if not item.get("instruction") or not item.get("code"):
                continue

            samples.append(item)

    return samples


def visible_text(text: str) -> str:
    return (
        text
        .replace(" ", "·")
        .replace("\t", "\\t")
        .replace("\n", "\\n\n")
    )


def token_preview(tokenizer, input_ids: List[int], limit: int = 120) -> str:
    tokens = tokenizer.convert_ids_to_tokens(input_ids[:limit])
    return " ".join(tokens)


def count_format_tokens(tokenizer, input_ids: List[int]) -> Dict[str, int]:
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    newline_count = 0
    space_like_count = 0
    indent_like_count = 0

    for token in tokens:
        if "Ċ" in token or "\n" in token:
            newline_count += 1

        if "Ġ" in token or token.startswith("▁") or token == " ":
            space_like_count += 1

        if "ĊĠ" in token or "ĊĊ" in token or token.count("Ġ") >= 4:
            indent_like_count += 1

    return {
        "newline_token_count": newline_count,
        "space_like_token_count": space_like_count,
        "indent_like_token_count": indent_like_count,
    }


def check_roundtrip_decode(tokenizer, sample: Dict[str, str], index: int) -> None:
    text = build_text(sample)

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    input_ids = encoded["input_ids"]

    decoded = tokenizer.decode(
        input_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    print("\n" + "=" * 80)
    print(f"[Roundtrip Check] sample #{index}")
    print("-" * 80)
    print(f"原始 text 长度: {len(text)}")
    print(f"token 数量: {len(input_ids)}")
    print(f"decoded 长度: {len(decoded)}")
    print(f"是否被截断: {len(input_ids) >= MAX_LENGTH}")
    print(f"decoded 是否包含换行: {'\\n' in decoded}")
    print(f"decoded 是否包含 4 空格缩进: {'    ' in decoded}")

    print("\n[原始 text 片段，可视化空格/换行]")
    print(visible_text(text[:800]))

    print("\n[decode 后片段，可视化空格/换行]")
    print(visible_text(decoded[:800]))

    print("\n[token preview]")
    print(token_preview(tokenizer, input_ids))

    print("\n[format token counts]")
    print(count_format_tokens(tokenizer, input_ids))


def build_labels_completion_only(
    tokenizer,
    sample: Dict[str, str],
) -> Tuple[List[int], List[int], List[int]]:
    prompt = build_prompt(sample)
    completion = str(sample["code"]).strip() + "\n```"

    full_text = prompt + completion

    full_encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    prompt_encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    input_ids = full_encoded["input_ids"]
    prompt_ids = prompt_encoded["input_ids"]

    labels = input_ids.copy()

    mismatch = input_ids[:len(prompt_ids)] != prompt_ids

    if mismatch:
        print("[WARNING] prompt token 与 full_text 开头 token 不一致，completion mask 可能错位")

    prompt_len = len(prompt_ids)

    for i in range(min(prompt_len, len(labels))):
        labels[i] = IGNORE_INDEX

    attention_mask = [1] * len(input_ids)

    return input_ids, labels, attention_mask


def check_labels_have_format_tokens(tokenizer, sample: Dict[str, str], index: int) -> None:
    input_ids, labels, attention_mask = build_labels_completion_only(tokenizer, sample)

    trained_token_ids = [
        token_id
        for token_id, label in zip(input_ids, labels)
        if label != IGNORE_INDEX
    ]

    trained_text = tokenizer.decode(
        trained_token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    print("\n" + "=" * 80)
    print(f"[SFT Labels Check] sample #{index}")
    print("-" * 80)
    print(f"input_ids 长度: {len(input_ids)}")
    print(f"labels 有效 token 数: {len(trained_token_ids)}")
    print(f"attention_mask 长度: {len(attention_mask)}")
    print(f"labels 中是否包含换行: {'\\n' in trained_text}")
    print(f"labels 中是否包含 4 空格缩进: {'    ' in trained_text}")

    print("\n[labels 参与训练的文本片段，可视化空格/换行]")
    print(visible_text(trained_text[:800]))

    print("\n[labels token preview]")
    print(token_preview(tokenizer, trained_token_ids))

    print("\n[labels format token counts]")
    print(count_format_tokens(tokenizer, trained_token_ids))


def main():
    tokenizer = load_tokenizer()
    samples = load_samples()

    print(f"[INFO] 加载样本数: {len(samples)}")
    print(f"[INFO] tokenizer: {MODEL_PATH}")
    print(f"[INFO] eos_token: {repr(tokenizer.eos_token)}")
    print(f"[INFO] pad_token: {repr(tokenizer.pad_token)}")

    if not samples:
        raise ValueError("没有可检查的 code_comment 样本")

    selected = random.sample(samples, min(SAMPLE_SIZE, len(samples)))

    for idx, sample in enumerate(selected, start=1):
        check_roundtrip_decode(tokenizer, sample, idx)
        check_labels_have_format_tokens(tokenizer, sample, idx)


if __name__ == "__main__":
    main()