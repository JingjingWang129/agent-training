import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from config.settings import LOG_DIR, MODEL_PATH


PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_SAMPLES_PATH = PROJECT_ROOT / "data" / "training_samples.jsonl"
TOKENIZED_DIR = PROJECT_ROOT / "data" / "tokenized"
LOGS_DIR = PROJECT_ROOT / LOG_DIR

SKIPPED_LOG = LOGS_DIR / "build_tokens_skipped.log"
TRUNCATED_LOG = LOGS_DIR / "build_tokens_truncated.log"
ERROR_LOG = LOGS_DIR / "build_tokens_errors.log"

MAX_LENGTH = 2048


class TokenBuilder:
    def __init__(self):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        TOKENIZED_DIR.parent.mkdir(parents=True, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_PATH),
            trust_remote_code=True,
        )

        if self.tokenizer.eos_token is None:
            raise ValueError("当前 tokenizer 没有 eos_token，请先检查模型 tokenizer 配置")

        self.stats = {
            "total_samples": 0,
            "total_tokens": 0,
            "max_tokens": 0,
            "min_tokens": None,
            "type_counts": Counter(),
            "skipped": 0,
            "truncated": 0,
            "errors": 0,
        }

    def _now(self) -> str:
        return datetime.now().isoformat(timespec="seconds")

    def _write_log(self, log_path: Path, message: str) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{self._now()}] {message}\n")

    def _extract_code(self, sample: Dict[str, Any]) -> str:
        sample_type = sample.get("type")

        if sample_type == "code_comment":
            return sample.get("code") or ""

        if sample_type == "completion":
            return sample.get("original_code") or sample.get("code") or ""

        if sample_type == "bug_fix":
            return sample.get("fixed_code") or sample.get("code") or ""

        return sample.get("code") or ""

    def _is_valid_sample(self, sample: Dict[str, Any]) -> bool:
        code = self._extract_code(sample)

        if not code or not code.strip():
            self.stats["skipped"] += 1
            self._write_log(
                SKIPPED_LOG,
                f"跳过空 code 样本: type={sample.get('type')}, instruction={sample.get('instruction', '')[:80]}",
            )
            return False

        return True

    def _tokenize_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        try:
            code = self._extract_code(sample)
            text = code + self.tokenizer.eos_token

            full_tokenized = self.tokenizer(
                text,
                truncation=False,
                padding=False,
                add_special_tokens=False,
            )
            full_input_ids = full_tokenized["input_ids"]

            if len(full_input_ids) > MAX_LENGTH:
                self.stats["truncated"] += 1
                self._write_log(
                    TRUNCATED_LOG,
                    f"样本被截断: type={sample.get('type')}, original_tokens={len(full_input_ids)}, max_length={MAX_LENGTH}",
                )

            tokenized = self.tokenizer(
                text,
                max_length=MAX_LENGTH,
                truncation=True,
                padding=False,
                add_special_tokens=False,
            )

            input_ids = tokenized["input_ids"]
            token_count = len(input_ids)

            self.stats["total_samples"] += 1
            self.stats["total_tokens"] += token_count
            self.stats["max_tokens"] = max(self.stats["max_tokens"], token_count)

            if self.stats["min_tokens"] is None:
                self.stats["min_tokens"] = token_count
            else:
                self.stats["min_tokens"] = min(self.stats["min_tokens"], token_count)

            sample_type = sample.get("type", "unknown")
            self.stats["type_counts"][sample_type] += 1

            result = {
                "type": sample_type,
                "input_ids": input_ids,
            }

            if sample.get("instruction"):
                result["instruction"] = sample["instruction"]

            return result

        except Exception as exc:
            self.stats["errors"] += 1
            self._write_log(ERROR_LOG, f"tokenization 失败: {exc}")
            return {
                "type": sample.get("type", "unknown"),
                "instruction": sample.get("instruction", ""),
                "input_ids": [],
            }

    def _load_dataset_normal(self):
        return load_dataset(
            "json",
            data_files=str(TRAINING_SAMPLES_PATH),
            split="train",
        )

    def _load_dataset_streaming(self):
        return load_dataset(
            "json",
            data_files=str(TRAINING_SAMPLES_PATH),
            split="train",
            streaming=True,
        )

    def _build_with_normal_dataset(self) -> Dataset:
        dataset = self._load_dataset_normal()

        dataset = dataset.filter(
            self._is_valid_sample,
            desc="过滤空 code 样本",
        )

        tokenized_dataset = dataset.map(
            self._tokenize_sample,
            batched=False,
            remove_columns=dataset.column_names,
            desc="Tokenizing samples",
        )

        tokenized_dataset = tokenized_dataset.filter(
            lambda sample: bool(sample.get("input_ids")),
            desc="过滤 tokenization 失败样本",
        )

        return tokenized_dataset

    def _build_with_streaming_dataset(self) -> Dataset:
        streaming_dataset = self._load_dataset_streaming()
        rows: List[Dict[str, Any]] = []

        for sample in streaming_dataset:
            try:
                if not self._is_valid_sample(sample):
                    continue

                tokenized_sample = self._tokenize_sample(sample)

                if tokenized_sample.get("input_ids"):
                    rows.append(tokenized_sample)

            except Exception as exc:
                self.stats["errors"] += 1
                self._write_log(ERROR_LOG, f"流式处理样本失败: {exc}")

        return Dataset.from_list(rows)

    def build(self) -> Dataset:
        if not TRAINING_SAMPLES_PATH.exists():
            raise FileNotFoundError(f"训练样本文件不存在: {TRAINING_SAMPLES_PATH}")

        try:
            tokenized_dataset = self._build_with_normal_dataset()
        except MemoryError:
            self._write_log(ERROR_LOG, "普通模式加载数据集内存不足，切换到 streaming=True")
            tokenized_dataset = self._build_with_streaming_dataset()
        except Exception as exc:
            self._write_log(ERROR_LOG, f"普通模式加载失败，切换到 streaming=True: {exc}")
            tokenized_dataset = self._build_with_streaming_dataset()

        tokenized_dataset.save_to_disk(str(TOKENIZED_DIR))
        return tokenized_dataset

    def print_summary(self) -> None:
        total_samples = self.stats["total_samples"]
        total_tokens = self.stats["total_tokens"]

        average_tokens = total_tokens / total_samples if total_samples else 0
        min_tokens = self.stats["min_tokens"] if self.stats["min_tokens"] is not None else 0

        print("\n========== Token Build Summary ==========")
        print(f"总样本数: {total_samples}")
        print(f"总 token 数: {total_tokens}")
        print(f"平均 token 长度: {average_tokens:.2f}")
        print(f"最大 token 长度: {self.stats['max_tokens']}")
        print(f"最小 token 长度: {min_tokens}")

        print("各类型样本数量:")
        for sample_type, count in self.stats["type_counts"].items():
            print(f"  - {sample_type}: {count}")

        print(f"跳过空 code 样本数: {self.stats['skipped']}")
        print(f"被截断样本数: {self.stats['truncated']}")
        print(f"错误样本数: {self.stats['errors']}")
        print(f"保存目录: {TOKENIZED_DIR}")
        print("========================================\n")


if __name__ == "__main__":
    print("开始构建 tokens...")

    builder = TokenBuilder()
    builder.build()
    builder.print_summary()

    print("构建完成！")