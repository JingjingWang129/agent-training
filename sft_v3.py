from pathlib import Path
from typing import Any, Dict

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from trl import SFTConfig, SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parent

# 对 deepseek-coder 初始模型进行 sft
MODEL_PATH = PROJECT_ROOT / "models" / "deepseek-coder-1.3b-base"
DATA_PATH = PROJECT_ROOT / "data" / "sft_samples_1000.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "new_sft_checkpoints"
FINAL_MODEL_DIR = PROJECT_ROOT / "new_sft_model"
LOG_DIR = PROJECT_ROOT / "logs"
ERROR_LOG = LOG_DIR / "sft_errors.log"


def log_error(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with ERROR_LOG.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def load_model_and_tokenizer():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"模型路径不存在: {MODEL_PATH}")

    print(f"[INFO] 加载 tokenizer: {MODEL_PATH}")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(MODEL_PATH / "tokenizer.json"),
        bos_token="<｜begin▁of▁sentence｜>",
        eos_token="<｜end▁of▁sentence｜>",
        pad_token="<｜end▁of▁sentence｜>",
        clean_up_tokenization_spaces=False,
        model_max_length=16384,
    )

    print(f"[INFO] 加载模型: {MODEL_PATH}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    return model, tokenizer


def format_prompt(instruction: str) -> str:
    return (
        "### System:\n"
        "You are a Python coding assistant.Use Python 3 syntax."
        "Write only the code required to fulfill the user's instruction. "
        "Stop after ## End of Code ##.\n\n"
        "### Instruction:\n"
        f"{instruction.strip()}\n\n"
        "### Response:\n"
    )


def prepare_dataset(tokenizer) -> Dataset:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"训练数据不存在: {DATA_PATH}")

    print(f"[INFO] 加载训练数据: {DATA_PATH}")

    raw_dataset = load_dataset(
        "json",
        data_files=str(DATA_PATH),
        split="train",
    )

    print(f"[INFO] 原始样本数: {len(raw_dataset)}")

    dataset = raw_dataset.filter(
        lambda x: x.get("type") == "code_comment",
        desc="过滤 code_comment 样本",
    )

    print(f"[INFO] code_comment 样本数: {len(dataset)}")

    eos_token = tokenizer.eos_token or ""

    def build_text(example: Dict[str, Any]) -> Dict[str, str]:
        instruction = str(example.get("instruction") or "").strip()
        code = str(example.get("code") or "").strip()

        if not instruction or not code:
            return {"text": ""}

        text = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Response:\n"
            "```python\n"
            f"{code}\n"
            "```"
            "\n### End of Code ###"
            f"{eos_token}"
        )
        return {"text": text}

    dataset = dataset.map(
        build_text,
        remove_columns=dataset.column_names,
        desc="构建 text 数据",
    )

    before_filter = len(dataset)

    dataset = dataset.filter(
        lambda x: bool(x["text"].strip()),
        desc="过滤空 text",
    )

    skipped = before_filter - len(dataset)
    if skipped:
        log_error(f"[WARNING] 跳过空样本数量: {skipped}")

    print(f"[INFO] 可训练样本数: {len(dataset)}")

    if len(dataset) == 0:
        raise ValueError("没有可用于 SFT 的 code_comment 样本，请检查 training_samples.jsonl")

    print(f"[DEBUG] Dataset columns: {dataset.column_names}")
    # 检查第一条数据的 completion_mask（如果存在）
    if 'completion_mask' in dataset.column_names:
        print(f"[DEBUG] completion_mask sample: {dataset[0]['completion_mask']}")

    return dataset


def build_training_args() -> SFTConfig:
    return SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        dataset_text_field="text",
        completion_only_loss=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=2048,
        packing=False,
        bf16=True,
        logging_steps=20,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=True,
    )


def print_config(dataset: Dataset, training_args: SFTConfig) -> None:
    print("\n========== 第三遍 SFT Config ==========")
    print(f"模型路径: {MODEL_PATH}")
    print(f"数据路径: {DATA_PATH}")
    print(f"输出路径: {FINAL_MODEL_DIR}")
    print(f"训练样本数: {len(dataset)}")
    print(f"epoch: {training_args.num_train_epochs}")
    print(f"batch_size: {training_args.per_device_train_batch_size}")
    print(f"gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
    print(f"learning_rate: {training_args.learning_rate}")
    print(f"warmup_ratio: {training_args.warmup_ratio}")
    print(f"max_length: {training_args.max_length}")
    print(f"bf16: {training_args.bf16}")
    print(f"save_steps: {training_args.save_steps}")
    print("=========================================\n")


def main() -> None:
    try:
        print("开始第三遍SFT...")
        print(f"将从 {MODEL_PATH} 继续训练")

        model, tokenizer = load_model_and_tokenizer()
        dataset = prepare_dataset(tokenizer)
        training_args = build_training_args()

        print_config(dataset, training_args)

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        trainer.train()

        FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(FINAL_MODEL_DIR))
        tokenizer.save_pretrained(str(FINAL_MODEL_DIR))

        print(f"训练完成！最终模型已保存到: {FINAL_MODEL_DIR}")

    except Exception as exc:
        log_error(f"[ERROR] SFT 训练失败: {repr(exc)}")
        raise


if __name__ == "__main__":
    main()