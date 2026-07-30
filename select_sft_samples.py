import ast
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_PATH = PROJECT_ROOT / "data" / "training_samples.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "sft_samples_1000.jsonl"
REPORT_PATH = PROJECT_ROOT / "data" / "sft_samples_1000_report.json"

TARGET_SIZE = 1000
RANDOM_SEED = 42


MIN_CODE_CHARS = 100
MAX_CODE_CHARS = 2000
MIN_INSTRUCTION_CHARS = 10
MAX_INSTRUCTION_CHARS = 160


BAD_INSTRUCTION_PATTERNS = [
    r"^\s*todo\b",
    r"^\s*fixme\b",
    r"^\s*test\b",
    r"^\s*deprecated\b",
    r"^\s*internal\b",
    r"实现 .* 模块的完整功能",
    r"实现 .* 模块的功能",
]

BAD_CODE_PATTERNS = [
    r"\bpytest\b",
    r"\bunittest\b",
    r"\bmock\b",
    r"\bMock\b",
    r"\bassert\b",
    r"\bTODO\b",
    r"\bFIXME\b",
    r"\bpass\s*(#.*)?$",
    r"NotImplementedError",
    r"raise\s+NotImplemented",
    r"if\s+__name__\s*==\s*['\"]__main__['\"]",
    r"argparse\.",
    r"click\.",
]


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError:
                yield line_no, None


def normalize_instruction(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_code(code: str) -> str:
    return (code or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def is_ast_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def get_code_shape(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "invalid"

    top_nodes = [
        node for node in tree.body
        if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    if len(top_nodes) == 1 and isinstance(top_nodes[0], ast.FunctionDef):
        return "function"

    if len(top_nodes) == 1 and isinstance(top_nodes[0], ast.AsyncFunctionDef):
        return "function"

    if len(top_nodes) == 1 and isinstance(top_nodes[0], ast.ClassDef):
        return "class"

    return "other"


def has_good_indentation(code: str) -> bool:
    return "\n    " in code or "\n\t" in code


def count_imports(code: str) -> int:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 999

    return sum(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))


def count_top_level_defs(code: str) -> int:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 999

    return sum(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in tree.body
    )


def contains_bad_pattern(text: str, patterns: List[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) for pattern in patterns)


def quality_score(sample: Dict[str, str]) -> int:
    instruction = sample["instruction"]
    code = sample["code"]
    shape = sample["shape"]

    score = 0

    if shape == "function":
        score += 40
    elif shape == "class":
        score += 30

    code_len = len(code)
    if 200 <= code_len <= 1200:
        score += 25
    elif 1200 < code_len <= 2000:
        score += 10

    instruction_len = len(instruction)
    if 20 <= instruction_len <= 120:
        score += 20
    elif 10 <= instruction_len < 20:
        score += 8

    if has_good_indentation(code):
        score += 15

    if count_imports(code) <= 2:
        score += 10

    if count_top_level_defs(code) == 1:
        score += 10

    if '"""' in code or "'''" in code:
        score += 5

    return score


def filter_sample(line_no: int, item: Optional[Dict]) -> tuple[Optional[Dict], Optional[str]]:
    if item is None:
        return None, "json_decode_error"

    if item.get("type") != "code_comment":
        return None, "not_code_comment"

    instruction = normalize_instruction(str(item.get("instruction") or ""))
    code = normalize_code(str(item.get("code") or ""))

    if not instruction:
        return None, "empty_instruction"

    if not code:
        return None, "empty_code"

    if len(instruction) < MIN_INSTRUCTION_CHARS:
        return None, "instruction_too_short"

    if len(instruction) > MAX_INSTRUCTION_CHARS:
        return None, "instruction_too_long"

    if len(code) < MIN_CODE_CHARS:
        return None, "code_too_short"

    if len(code) > MAX_CODE_CHARS:
        return None, "code_too_long"

    if not has_good_indentation(code):
        return None, "no_indentation"

    if contains_bad_pattern(instruction, BAD_INSTRUCTION_PATTERNS):
        return None, "bad_instruction_pattern"

    if contains_bad_pattern(code, BAD_CODE_PATTERNS):
        return None, "bad_code_pattern"

    if not is_ast_valid(code):
        return None, "invalid_python"

    shape = get_code_shape(code)
    if shape not in {"function", "class"}:
        return None, "not_function_or_class"

    if count_imports(code) > 3:
        return None, "too_many_imports"

    if count_top_level_defs(code) != 1:
        return None, "multiple_top_level_defs"

    cleaned = {
        "type": "code_comment",
        "instruction": instruction,
        "code": code,
        "shape": shape,
        "source_line": line_no,
    }

    cleaned["score"] = quality_score(cleaned)
    return cleaned, None


def deduplicate(samples: List[Dict]) -> List[Dict]:
    seen_code = set()
    seen_pair = set()
    result = []

    for sample in samples:
        code_key = sample["code"]
        pair_key = (sample["instruction"].lower(), sample["code"])

        if code_key in seen_code:
            continue

        if pair_key in seen_pair:
            continue

        seen_code.add(code_key)
        seen_pair.add(pair_key)
        result.append(sample)

    return result


def balanced_select(samples: List[Dict], target_size: int) -> List[Dict]:
    functions = [s for s in samples if s["shape"] == "function"]
    classes = [s for s in samples if s["shape"] == "class"]

    functions.sort(key=lambda x: x["score"], reverse=True)
    classes.sort(key=lambda x: x["score"], reverse=True)

    target_classes = min(len(classes), int(target_size * 0.2))
    target_functions = target_size - target_classes

    selected = functions[:target_functions] + classes[:target_classes]

    if len(selected) < target_size:
        selected_keys = {id(s) for s in selected}
        remaining = [s for s in samples if id(s) not in selected_keys]
        remaining.sort(key=lambda x: x["score"], reverse=True)
        selected.extend(remaining[:target_size - len(selected)])

    random.Random(RANDOM_SEED).shuffle(selected)
    return selected[:target_size]


def write_jsonl(samples: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            output = {
                "type": sample["type"],
                "instruction": sample["instruction"],
                "code": sample["code"],
            }
            f.write(json.dumps(output, ensure_ascii=False) + "\n")


def write_report(
    total_seen: int,
    filtered_count: int,
    selected: List[Dict],
    skip_reasons: Counter,
) -> None:
    report = {
        "input_path": str(INPUT_PATH),
        "output_path": str(OUTPUT_PATH),
        "total_seen": total_seen,
        "filtered_count": filtered_count,
        "selected_count": len(selected),
        "shape_counts": dict(Counter(sample["shape"] for sample in selected)),
        "score_min": min((sample["score"] for sample in selected), default=0),
        "score_max": max((sample["score"] for sample in selected), default=0),
        "score_avg": (
            sum(sample["score"] for sample in selected) / len(selected)
            if selected else 0
        ),
        "skip_reasons": dict(skip_reasons),
    }

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"找不到输入文件: {INPUT_PATH}")

    random.seed(RANDOM_SEED)

    total_seen = 0
    candidates = []
    skip_reasons = Counter()

    for line_no, item in read_jsonl(INPUT_PATH):
        total_seen += 1

        sample, reason = filter_sample(line_no, item)

        if sample is None:
            skip_reasons[reason] += 1
            continue

        candidates.append(sample)

    candidates = deduplicate(candidates)
    candidates.sort(key=lambda x: x["score"], reverse=True)

    selected = balanced_select(candidates, TARGET_SIZE)

    write_jsonl(selected, OUTPUT_PATH)
    write_report(total_seen, len(candidates), selected, skip_reasons)

    print("========== SFT Sample Selection ==========")
    print(f"输入文件: {INPUT_PATH}")
    print(f"输出文件: {OUTPUT_PATH}")
    print(f"报告文件: {REPORT_PATH}")
    print(f"读取样本数: {total_seen}")
    print(f"过滤后候选数: {len(candidates)}")
    print(f"最终选择数: {len(selected)}")
    print(f"类型分布: {dict(Counter(sample['shape'] for sample in selected))}")
    print("跳过原因:")
    for reason, count in skip_reasons.most_common():
        print(f"  - {reason}: {count}")
    print("==========================================")


if __name__ == "__main__":
    main()