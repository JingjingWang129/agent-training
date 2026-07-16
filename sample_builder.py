import ast
import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import LOG_DIR


PROJECT_ROOT = Path(__file__).resolve().parent
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
MANIFEST_PATH = CLEANED_DIR / "manifest.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "training_samples.jsonl"
LOGS_DIR = PROJECT_ROOT / LOG_DIR
ERROR_LOG = LOGS_DIR / "sample_builder_errors.log"

MAX_SAMPLE_LENGTH = 4096
MASK_TOKEN = "<MASK>"


class SampleBuilder:
    def __init__(self, random_seed: int = 42):
        self.random = random.Random(random_seed)
        self.seen_hashes = set()
        self.manifest = []
        self.stats = {
            "total_files": 0,
            "generated_by_type": Counter(),
            "written_by_type": Counter(),
            "skip_reasons": Counter(),
        }

        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    def _now(self) -> str:
        return datetime.now().isoformat(timespec="seconds")

    def _log_error(self, message: str) -> None:
        with ERROR_LOG.open("a", encoding="utf-8") as f:
            f.write(f"[{self._now()}] {message}\n")

    def _load_manifest(self) -> None:
        if not MANIFEST_PATH.exists():
            self._log_error(f"[WARNING] manifest.json 不存在: {MANIFEST_PATH}")
            self.manifest = []
            return

        try:
            with MANIFEST_PATH.open("r", encoding="utf-8") as f:
                self.manifest = json.load(f)
        except Exception as exc:
            self._log_error(f"[ERROR] 读取 manifest.json 失败: {exc}")
            self.manifest = []

    def _read_file(self, file_path: Path) -> Optional[str]:
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                return file_path.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                self._log_error(f"[ERROR] 读取文件失败 {file_path}: {exc}")
                return None
        except Exception as exc:
            self._log_error(f"[ERROR] 读取文件失败 {file_path}: {exc}")
            return None

    def _parse_ast(self, source: str, file_path: Path) -> Optional[ast.AST]:
        try:
            return ast.parse(source)
        except SyntaxError as exc:
            self.stats["skip_reasons"]["ast_parse_failed"] += 1
            self._log_error(f"[ERROR] AST 解析失败 {file_path}: {exc}")
            return None
        except Exception as exc:
            self.stats["skip_reasons"]["ast_parse_failed"] += 1
            self._log_error(f"[ERROR] AST 解析异常 {file_path}: {exc}")
            return None

    def _extract_above_comments(self, source_lines: List[str], lineno: int) -> str:
        comments = []
        index = lineno - 2

        while index >= 0:
            line = source_lines[index].strip()

            if not line:
                index -= 1
                continue

            if line.startswith("#"):
                comment = line.lstrip("#").strip()
                if comment:
                    comments.append(comment)
                index -= 1
                continue

            break

        comments.reverse()
        return " ".join(comments).strip()

    def _instruction_from_signature(self, node: ast.FunctionDef) -> str:
        args = [arg.arg for arg in node.args.args]

        if node.args.vararg:
            args.append("*" + node.args.vararg.arg)

        if node.args.kwarg:
            args.append("**" + node.args.kwarg.arg)

        readable_name = node.name.replace("_", " ")

        if args:
            return f"实现函数 {readable_name}，参数包括 {', '.join(args)}"

        return f"实现函数 {readable_name}"

    def _extract_function_samples(
        self,
        source: str,
        tree: ast.AST,
    ) -> List[Dict[str, str]]:
        samples = []
        source_lines = source.splitlines()

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue

            function_code = ast.get_source_segment(source, node)
            if not function_code:
                self.stats["skip_reasons"]["empty_function_code"] += 1
                continue

            docstring = ast.get_docstring(node)
            above_comments = self._extract_above_comments(source_lines, node.lineno)

            instruction = docstring or above_comments or self._instruction_from_signature(node)
            instruction = re.sub(r"\s+", " ", instruction).strip()

            samples.append({
                "type": "code_comment",
                "instruction": instruction,
                "code": function_code.strip(),
            })

        return samples

    def _build_completion_sample(self, source: str) -> Optional[Dict[str, str]]:
        lines = source.splitlines()

        if not lines:
            return None

        candidate_indexes = [
            i for i, line in enumerate(lines)
            if line.strip() and line.strip() != MASK_TOKEN
        ]

        if not candidate_indexes:
            return None

        mask_count = max(1, int(len(candidate_indexes) * 0.2))
        mask_count = min(mask_count, len(candidate_indexes))

        selected_indexes = set(self.random.sample(candidate_indexes, mask_count))
        masked_lines = []

        for i, line in enumerate(lines):
            if i in selected_indexes:
                indent = line[:len(line) - len(line.lstrip())]
                masked_lines.append(f"{indent}{MASK_TOKEN}")
            else:
                masked_lines.append(line)

        return {
            "type": "completion",
            "masked_code": "\n".join(masked_lines),
            "original_code": source,
        }

    def _simple_bug_candidates(self, lines: List[str]) -> List[Dict[str, Any]]:
        candidates = []

        for index, line in enumerate(lines):
            stripped = line.strip()

            if not stripped:
                continue

            if stripped.startswith((
                "def ",
                "class ",
                "import ",
                "from ",
                "try:",
                "except",
                "finally:",
                "with ",
                "for ",
                "while ",
                "if ",
                "elif ",
                "else:",
            )):
                continue

            if len(stripped) > 120:
                continue

            if " + " in line:
                candidates.append({"index": index, "old": " + ", "new": " - "})

            if "True" in line:
                candidates.append({"index": index, "old": "True", "new": "False"})

            if "False" in line:
                candidates.append({"index": index, "old": "False", "new": "True"})

            if " == " in line:
                candidates.append({"index": index, "old": " == ", "new": " != "})

            if ")" in line and "(" in line and line.count("(") == line.count(")"):
                candidates.append({"index": index, "delete_char": ")"})

        return candidates

    def _build_bug_fix_sample(self, source: str) -> Optional[Dict[str, str]]:
        lines = source.splitlines()
        candidates = self._simple_bug_candidates(lines)

        if not candidates:
            return None

        bug_count = min(len(candidates), self.random.randint(1, 2))
        selected = self.random.sample(candidates, bug_count)

        buggy_lines = lines[:]

        for item in selected:
            index = item["index"]

            if "delete_char" in item:
                char = item["delete_char"]
                buggy_lines[index] = buggy_lines[index].replace(char, "", 1)
            else:
                buggy_lines[index] = buggy_lines[index].replace(
                    item["old"],
                    item["new"],
                    1,
                )

        buggy_code = "\n".join(buggy_lines)

        if buggy_code == source:
            return None

        return {
            "type": "bug_fix",
            "buggy_code": buggy_code,
            "fixed_code": source,
        }

    def _get_code_for_hash(self, sample: Dict[str, str]) -> str:
        sample_type = sample.get("type")

        if sample_type == "code_comment":
            return sample.get("code", "")

        if sample_type == "completion":
            return sample.get("original_code", "")

        if sample_type == "bug_fix":
            return sample.get("fixed_code", "")

        return ""

    def _hash_code(self, code: str) -> str:
        return hashlib.md5(code.encode("utf-8")).hexdigest()

    def _json_length(self, sample: Dict[str, str]) -> int:
        return len(json.dumps(sample, ensure_ascii=False))

    def _truncate_sample(self, sample: Dict[str, str], file_path: Path) -> Dict[str, str]:
        if self._json_length(sample) <= MAX_SAMPLE_LENGTH:
            return sample

        self.stats["skip_reasons"]["truncated_too_long"] += 1
        self._log_error(f"[WARNING] 样本超过 {MAX_SAMPLE_LENGTH} 字符，已截断: {file_path}")

        result = dict(sample)
        priority_fields = [
            "code",
            "original_code",
            "masked_code",
            "fixed_code",
            "buggy_code",
        ]

        while self._json_length(result) > MAX_SAMPLE_LENGTH:
            longest_field = None
            longest_length = 0

            for field in priority_fields:
                value = result.get(field)
                if isinstance(value, str) and len(value) > longest_length:
                    longest_field = field
                    longest_length = len(value)

            if not longest_field or longest_length <= 20:
                break

            keep_length = max(20, int(longest_length * 0.85))
            result[longest_field] = result[longest_field][:keep_length].rstrip()

        return result

    def _validate_sample(self, sample: Dict[str, str]) -> bool:
        sample_type = sample.get("type")
        code = self._get_code_for_hash(sample)

        if not code or len(code.strip()) < 20:
            self.stats["skip_reasons"]["code_too_short"] += 1
            return False

        if sample_type == "code_comment":
            instruction = sample.get("instruction", "").strip()
            if len(instruction) < 5:
                self.stats["skip_reasons"]["instruction_too_short"] += 1
                return False

        return True

    def _write_sample(self, output_file, sample: Dict[str, str], file_path: Path) -> bool:
        sample = self._truncate_sample(sample, file_path)

        if not self._validate_sample(sample):
            return False

        code = self._get_code_for_hash(sample)
        code_hash = self._hash_code(code)

        if code_hash in self.seen_hashes:
            self.stats["skip_reasons"]["duplicate"] += 1
            return False

        self.seen_hashes.add(code_hash)

        output_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
        self.stats["written_by_type"][sample["type"]] += 1
        return True

    def _build_samples_for_file(self, file_path: Path) -> List[Dict[str, str]]:
        source = self._read_file(file_path)
        if not source:
            self.stats["skip_reasons"]["empty_or_unreadable_file"] += 1
            return []

        tree = self._parse_ast(source, file_path)
        if tree is None:
            return []

        samples = []

        try:
            samples.extend(self._extract_function_samples(source, tree))

            completion_sample = self._build_completion_sample(source)
            if completion_sample:
                samples.append(completion_sample)

            bug_fix_sample = self._build_bug_fix_sample(source)
            if bug_fix_sample:
                samples.append(bug_fix_sample)

        except Exception as exc:
            self.stats["skip_reasons"]["sample_build_failed"] += 1
            self._log_error(f"[ERROR] 构建样本失败 {file_path}: {exc}")

        return samples

    def run(self) -> None:
        self._load_manifest()

        py_files = sorted(CLEANED_DIR.rglob("*.py"))
        self.stats["total_files"] = len(py_files)

        with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
            for index, file_path in enumerate(py_files, start=1):
                try:
                    samples = self._build_samples_for_file(file_path)

                    for sample in samples:
                        sample_type = sample.get("type", "unknown")
                        self.stats["generated_by_type"][sample_type] += 1
                        self._write_sample(output_file, sample, file_path)

                except Exception as exc:
                    self.stats["skip_reasons"]["file_processing_failed"] += 1
                    self._log_error(f"[ERROR] 处理文件失败 {file_path}: {exc}")

                if index % 100 == 0:
                    print(f"[PROGRESS] 已处理 {index}/{len(py_files)} 个文件")

    def print_summary(self) -> None:
        total_generated = sum(self.stats["generated_by_type"].values())
        total_written = sum(self.stats["written_by_type"].values())

        print("\n========== Sample Builder Summary ==========")
        print(f"总文件数: {self.stats['total_files']}")
        print(f"生成样本数: {total_generated}")
        print("生成样本数（分类型）:")
        for sample_type, count in self.stats["generated_by_type"].items():
            print(f"  - {sample_type}: {count}")

        print(f"去重后样本数: {total_written}")
        print("去重后样本数（分类型）:")
        for sample_type, count in self.stats["written_by_type"].items():
            print(f"  - {sample_type}: {count}")

        print("跳过原因统计:")
        for reason, count in self.stats["skip_reasons"].items():
            print(f"  - {reason}: {count}")

        print(f"输出文件: {OUTPUT_PATH}")
        print(f"错误日志: {ERROR_LOG}")
        print("===========================================\n")


if __name__ == "__main__":
    files = list(CLEANED_DIR.rglob("*.py"))
    print(f"[INFO] 扫描到 {len(files)} 个 cleaned Python 文件")

    builder = SampleBuilder()
    builder.run()
    builder.print_summary()

