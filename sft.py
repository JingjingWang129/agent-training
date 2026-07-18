from pathlib import Path
from typing import Any, Dict

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "deepseek-coder-1.3b-base"
DATA_PATH = PROJECT_ROOT / "data" / "training_samples.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "sft_checkpoints"
FINAL_MODEL_DIR = PROJECT_ROOT / "sft_model"
LOG_DIR = PROJECT_ROOT / "logs"
ERROR_LOG = LOG_DIR / "sft_errors.log"

MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 4


def log_error(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with ERROR_LOG.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def load_model_and_tokenizer():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"模型路径不存在: {MODEL_PATH}")

    print(f"[INFO] 加载 tokenizer: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        "### Instruction:\n"
        f"{instruction.strip()}\n\n"
        "### Response:\n"
    )


def prepare_dataset() -> Dataset:
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

    def build_prompt_completion(example: Dict[str, Any]) -> Dict[str, str]:
        instruction = str(example.get("instruction") or "").strip()
        code = str(example.get("code") or "").strip()

        if not instruction or not code:
            return {
                "prompt": "",
                "completion": "",
            }

        return {
            "prompt": format_prompt(instruction),
            "completion": code + "\n",
        }

    dataset = dataset.map(
        build_prompt_completion,
        remove_columns=dataset.column_names,
        desc="构建 prompt/completion 数据",
    )

    before_filter = len(dataset)

    dataset = dataset.filter(
        lambda x: bool(x["prompt"].strip()) and bool(x["completion"].strip()),
        desc="过滤空 prompt/completion",
    )

    skipped = before_filter - len(dataset)
    if skipped:
        log_error(f"[WARNING] 跳过空样本数量: {skipped}")

    print(f"[INFO] 可训练样本数: {len(dataset)}")

    if len(dataset) == 0:
        raise ValueError("没有可用于 SFT 的 code_comment 样本，请检查 training_samples.jsonl")

    return dataset


def build_training_args() -> SFTConfig:
    return SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        bf16=True,
        logging_steps=20,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        report_to="none",
        completion_only_loss=True,
        dataset_text_field="text",
        remove_unused_columns=True,
    )


def print_config(dataset: Dataset, training_args: SFTConfig) -> None:
    print("\n========== SFT Config ==========")
    print(f"模型路径: {MODEL_PATH}")
    print(f"数据路径: {DATA_PATH}")
    print(f"输出路径: {FINAL_MODEL_DIR}")
    print(f"训练样本数: {len(dataset)}")
    print(f"epoch: {training_args.num_train_epochs}")
    print(f"batch_size: {training_args.per_device_train_batch_size}")
    print(f"gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
    print(f"learning_rate: {training_args.learning_rate}")
    print(f"max_seq_length: {MAX_SEQ_LENGTH}")
    print(f"bf16: {training_args.bf16}")
    print(f"save_steps: {training_args.save_steps}")
    print("================================\n")


def main() -> None:
    try:
        print("开始 SFT 微调...")

        model, tokenizer = load_model_and_tokenizer()
        dataset = prepare_dataset()
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