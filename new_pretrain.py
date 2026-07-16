from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from config.settings import MODEL_PATH, PROJECT_ROOT


TOKENIZED_DATA_DIR = Path(PROJECT_ROOT) / "data" / "tokenized"
OUTPUT_DIR = Path(PROJECT_ROOT) / "checkpoints"
FINAL_MODEL_DIR = Path(PROJECT_ROOT) / "final_model"

MAX_SEQ_LENGTH = 2048


def get_torch_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16

    if torch.cuda.is_available():
        return torch.float16

    return torch.float32


def prepare_tokenized_dataset(dataset):
    columns_to_remove = [
        column for column in dataset.column_names
        if column not in ["input_ids", "attention_mask"]
    ]

    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)

    def ensure_attention_mask(example):
        input_ids = example.get("input_ids") or []
        attention_mask = example.get("attention_mask")

        if attention_mask is None or len(attention_mask) != len(input_ids):
            attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    dataset = dataset.map(
        ensure_attention_mask,
        batched=False,
        desc="整理 input_ids 和 attention_mask",
    )

    dataset = dataset.filter(
        lambda example: bool(example["input_ids"]),
        desc="过滤空样本",
    )

    return dataset


def print_training_info(dataset, training_args):
    token_lengths = [len(item) for item in dataset["input_ids"]]
    total_tokens = sum(token_lengths)

    print("========== Pretrain Config ==========")
    print(f"模型路径: {MODEL_PATH}")
    print(f"数据路径: {TOKENIZED_DATA_DIR}")
    print(f"样本数量: {len(dataset)}")
    print(f"总 token 数: {total_tokens}")
    print(f"最大长度: {max(token_lengths) if token_lengths else 0}")
    print(f"最小长度: {min(token_lengths) if token_lengths else 0}")
    print(f"平均长度: {total_tokens / len(token_lengths):.2f}" if token_lengths else "平均长度: 0")
    print(f"output_dir: {training_args.output_dir}")
    print(f"num_train_epochs: {training_args.num_train_epochs}")
    print(f"per_device_train_batch_size: {training_args.per_device_train_batch_size}")
    print(f"gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
    print(f"learning_rate: {training_args.learning_rate}")
    print(f"max_seq_length: {training_args.max_seq_length}")
    print(f"packing: {training_args.packing}")
    print(f"bf16: {training_args.bf16}")
    print(f"fp16: {training_args.fp16}")
    print("====================================\n")


def main():
    model_path = Path(MODEL_PATH)

    if not model_path.exists():
        raise FileNotFoundError(f"本地模型路径不存在: {model_path}")

    if not TOKENIZED_DATA_DIR.exists():
        raise FileNotFoundError(f"Tokenized 数据目录不存在: {TOKENIZED_DATA_DIR}")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    torch_dtype = get_torch_dtype()

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    dataset = load_from_disk(str(TOKENIZED_DATA_DIR))
    dataset = prepare_tokenized_dataset(dataset)

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=50,
        save_steps=1000,
        save_total_limit=3,
        report_to="none",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        bf16=use_bf16,
        fp16=use_fp16,
        remove_unused_columns=False,
        dataset_kwargs={
            "skip_prepare_dataset": True,
        },
    )

    print_training_info(dataset, training_args)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field=None,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
    )

    trainer.train()

    trainer.save_model(str(FINAL_MODEL_DIR))
    tokenizer.save_pretrained(str(FINAL_MODEL_DIR))

    print(f"最终模型已保存到: {FINAL_MODEL_DIR}")


if __name__ == "__main__":
    print("开始继续预训练...")
    main()
    print("训练完成！")