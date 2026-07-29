from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parent
# MODEL_PATH = PROJECT_ROOT / "final_model"
# MODEL_PATH = PROJECT_ROOT / "sft_model_v2"
MODEL_PATH = PROJECT_ROOT / "models" / "deepseek-coder-1.3b-base"


# ========tokenizer功能检查========
# tokenizer = AutoTokenizer.from_pretrained(
#     str(MODEL_PATH),
#     trust_remote_code=True,
#     use_fast=True,
# )

# text = "def foo(x: int) -> int:\n    return x + 1\n"

# ids = tokenizer(
#     text,
#     add_special_tokens=False,
# )["input_ids"]

# decoded = tokenizer.decode(
#     ids,
#     skip_special_tokens=False,
#     clean_up_tokenization_spaces=False,
# )

# print("TEXT REPR:", repr(text))
# print("DECODED REPR:", repr(decoded))
# print("EQUAL:", text == decoded)

# tokens = tokenizer.convert_ids_to_tokens(ids)
# for i, (tid, tok) in enumerate(zip(ids, tokens)):
#     print(i, tid, repr(tok), repr(tokenizer.decode([tid], clean_up_tokenization_spaces=False)))

# print(tokenizer)
# print(tokenizer.__class__)
# print(tokenizer.vocab_size)
# print(tokenizer.special_tokens_map)

# if hasattr(tokenizer, "backend_tokenizer"):
#     print(tokenizer.backend_tokenizer.pre_tokenizer)
#     print(tokenizer.backend_tokenizer.decoder)

# ========tokenizer配置检查========
# tokenizer_fast = AutoTokenizer.from_pretrained(
#     str(MODEL_PATH),
#     trust_remote_code=True,
#     use_fast=True,
# )

# tokenizer_slow = AutoTokenizer.from_pretrained(
#     str(MODEL_PATH),
#     trust_remote_code=True,
#     use_fast=False,
# )

# print("fast:", type(tokenizer_fast))
# print("slow:", type(tokenizer_slow))
# print("fast is_fast:", tokenizer_fast.is_fast)

# print("tokenizer_file:", tokenizer_fast.init_kwargs.get("tokenizer_file"))
# print("vocab_file:", tokenizer_fast.init_kwargs.get("vocab_file"))

# print("all files:")
# for p in MODEL_PATH.iterdir():
#     if "tokenizer" in p.name or p.name in ["special_tokens_map.json", "added_tokens.json"]:
#         print(p.name, p.stat().st_size)

# ========whitespace诊断========
# tests = [
#     " ",
#     "  ",
#     "\n",
#     "\n    ",
#     "def",
#     " def",
#     "\ndef",
#     "\n    return",
# ]

# for text in tests:
#     ids = tokenizer.encode(text, add_special_tokens=False)
#     toks = tokenizer.convert_ids_to_tokens(ids)
#     dec = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

#     print("=" * 40)
#     print("TEXT:", repr(text))
#     print("IDS:", ids)
#     print("TOKENS:", [repr(t) for t in toks])
#     print("DECODED:", repr(dec))

# ========检查 tokenizer.json========
# text = "def foo(x: int) -> int:\n    return x + 1\n"
# backend = Tokenizer.from_file(str(MODEL_PATH / "tokenizer.json"))

# enc = backend.encode(text)
# decoded = backend.decode(enc.ids)

# print("BACKEND PRE_TOKENIZER:", backend.pre_tokenizer)
# print("BACKEND DECODER:", backend.decoder)
# print("IDS:", enc.ids)
# print("TOKENS:", enc.tokens)
# print("DECODED REPR:", repr(decoded))
# print("EQUAL:", decoded == text)

# ========更换tokenizer加载路径========
# tokenizer.json本身没有受损，是版本与transformer v5的AutoTokenizer/LlamaTokenizer不兼容
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=str(MODEL_PATH / "tokenizer.json"),
    bos_token="<｜begin▁of▁sentence｜>",
    eos_token="<｜end▁of▁sentence｜>",
    pad_token="<｜end▁of▁sentence｜>",
    clean_up_tokenization_spaces=False,
    model_max_length=16384,
)

text = "def foo(x: int) -> int:\n    return x + 1\n"
ids = tokenizer.encode(text, add_special_tokens=False)
decoded = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

print("TEXT REPR:", repr(text))
print("DECODED REPR:", repr(decoded))
print("EQUAL:", text == decoded)
print(tokenizer.backend_tokenizer.pre_tokenizer)