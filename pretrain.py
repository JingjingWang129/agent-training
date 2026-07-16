import inspect
from pathlib import Path
from typing import Any, Dict

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from trl import SFTConfig, SFTTrainer

from config.settings import MODEL_PATH, PROJECT_ROOT


TOKENIZED_DATA_DIR = Path(PROJECT_ROOT) / "data" / "tokenized"
OUTPUT_DIR = Path(PROJECT_ROOT) / "checkpoints"
FINAL_MODEL_DIR = Path(PROJECT_ROOT) / "final_model"

MAX_SEQ_LENGTH = 2048


def build_sft_config() -> SFTConfig:
    base_args: Dict[str, Any] = {
        "output_dir": str(OUTPUT_DIR),
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 5e-5,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        "packing": True,
        "logging_steps": 50,
        "save_steps": 1000,
        "save_total_limit": 3,
        "report_to": "none",
        "remove_unused_columns": False,
    }

    signature = inspect.signature(SFTConfig.__init__)

    if "max_seq_length" in signature.parameters:
        base_args["max_seq_length"] = MAX_SEQ_LENGTH
    else:
        base_args["max_length"] = MAX_SEQ_LENGTH

    if "dataset_kwargs" in signature.parameters:
        base_args["dataset_kwargs"] = {"skip_prepare_dataset": True}

    return SFTConfig(**base_args)


def prepare_dataset(dataset):
    def add_training_columns(example):
        input_ids = example.get("input_ids")

        if input_ids is None:
            input_ids = []

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": input_ids.copy(),
        }

    keep_columns = {"input_ids", "attention_mask", "labels", "type", "instruction"}
    remove_columns = [
        column for column in dataset.column_names
        if column not in keep_columns
    ]

    dataset = dataset.map(
        add_training_columns,
        batched=False,
        remove_columns=remove_columns,
        desc="Preparing tokenized dataset",
    )

    dataset = dataset.filter(
        lambda example: bool(example.get("input_ids")),
        desc="Filtering empty tokenized samples",
    )

    return dataset


def print_training_info(dataset, training_args: SFTConfig) -> None:
    type_counts = {}

    if "type" in dataset.column_names:
        for sample_type in dataset["type"]:
            type_counts[sample_type] = type_counts.get(sample_type, 0) + 1

    print("========== Pretrain Config ==========")
    print(f"模型路径: {MODEL_PATH}")
    print(f"数据路径: {TOKENIZED_DATA_DIR}")
    print(f"数据总量: {len(dataset)}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"最终模型目录: {FINAL_MODEL_DIR}")
    print(f"epoch: {training_args.num_train_epochs}")
    print(f"batch_size: {training_args.per_device_train_batch_size}")
    print(f"gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
    print(f"learning_rate: {training_args.learning_rate}")
    print(f"lr_scheduler_type: {training_args.lr_scheduler_type}")
    print(f"warmup_ratio: {training_args.warmup_ratio}")
    print(f"bf16: {training_args.bf16}")
    print(f"packing: {training_args.packing}")
    print(f"max_seq_length: {MAX_SEQ_LENGTH}")

    if type_counts:
        print("样本类型统计:")
        for sample_type, count in type_counts.items():
            print(f"  - {sample_type}: {count}")

    print("====================================\n")


def main() -> None:
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"本地模型路径不存在: {MODEL_PATH}")

    if not TOKENIZED_DATA_DIR.exists():
        raise FileNotFoundError(f"tokenized 数据目录不存在: {TOKENIZED_DATA_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    dataset = load_from_disk(str(TOKENIZED_DATA_DIR))
    dataset = prepare_dataset(dataset)

    training_args = build_sft_config()
    print_training_info(dataset, training_args)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(str(FINAL_MODEL_DIR))
    tokenizer.save_pretrained(str(FINAL_MODEL_DIR))

    print(f"最终模型已保存到: {FINAL_MODEL_DIR}")


if __name__ == "__main__":
    print("开始继续预训练...")
    main()
    print("训练完成！")