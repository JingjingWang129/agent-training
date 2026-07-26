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


class SampleBuilder:
    """
    核心改进：
    1. 只生成 code_comment 类型（统一格式）
    2. 提取完整代码（完整函数/类/文件）
    3. 生成更自然的指令
    4. 自动过滤低质量样本
    """

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
        """加载 manifest.json"""
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
        """安全读取文件内容"""
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
        """解析 Python 代码为 AST"""
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
        """提取函数上方的注释作为指令候选"""
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

    def _generate_natural_instruction(self, node: ast.FunctionDef) -> str:
        """
        生成更自然的指令

        改进点：
        1. 根据函数名前缀推断意图
        2. 使用更丰富的模板
        3. 包含参数类型信息（如果可用）
        """
        name = node.name.replace("_", " ")

        # 提取参数名
        args = [arg.arg for arg in node.args.args]
        arg_names = ", ".join(args) if args else ""

        # 根据函数名前缀推断意图
        intent_templates = {
            "get": "获取 {rest}",
            "set": "设置 {rest}",
            "is_": "判断是否为 {rest}",
            "has_": "检查是否包含 {rest}",
            "calc": "计算 {rest}",
            "find": "查找 {rest}",
            "parse": "解析 {rest}",
            "validate": "验证 {rest}",
            "generate": "生成 {rest}",
            "create": "创建 {rest}",
            "init": "初始化 {rest}",
            "clean": "清理 {rest}",
            "process": "处理 {rest}",
            "load": "加载 {rest}",
            "save": "保存 {rest}",
            "read": "读取 {rest}",
            "write": "写入 {rest}",
            "open": "打开 {rest}",
            "close": "关闭 {rest}",
            "start": "启动 {rest}",
            "stop": "停止 {rest}",
            "run": "运行 {rest}",
            "test": "测试 {rest}",
        }

        # 尝试匹配前缀
        for prefix, template in intent_templates.items():
            if node.name.startswith(prefix):
                rest = node.name[len(prefix):].replace("_", " ")
                if rest:
                    instruction = template.format(rest=rest)
                else:
                    instruction = template.format(rest=name)

                if arg_names:
                    instruction += f"，参数为 {arg_names}"
                return instruction

        # 默认模板
        if args:
            return f"实现函数 {name}，接收参数 {arg_names}"
        else:
            return f"实现函数 {name}，无需参数"

    def _extract_function_samples(
            self,
            source: str,
            tree: ast.AST,
    ) -> List[Dict[str, str]]:
        """
        提取函数定义作为训练样本

        改进：每个函数作为一个独立样本，但确保代码完整
        """
        samples = []
        source_lines = source.splitlines()

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue

            # 提取完整的函数代码（包括装饰器）
            function_code = ast.get_source_segment(source, node)
            if not function_code:
                self.stats["skip_reasons"]["empty_function_code"] += 1
                continue

            # 优先使用 docstring，其次使用上方注释，最后从签名生成
            docstring = ast.get_docstring(node)
            above_comments = self._extract_above_comments(source_lines, node.lineno)

            if docstring:
                instruction = docstring
            elif above_comments:
                instruction = above_comments
            else:
                instruction = self._generate_natural_instruction(node)

            # 清理多余空白
            instruction = re.sub(r"\s+", " ", instruction).strip()

            # 确保指令以中文或英文开头，且长度合理
            if len(instruction) < 5:
                instruction = self._generate_natural_instruction(node)

            samples.append({
                "type": "code_comment",
                "instruction": instruction,
                "code": function_code.strip(),
            })

        return samples

    def _extract_class_samples(
            self,
            source: str,
            tree: ast.AST,
    ) -> List[Dict[str, str]]:
        """
        提取类定义作为训练样本

        新增功能：提取完整的类定义
        """
        samples = []
        source_lines = source.splitlines()

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            # 提取完整的类代码
            class_code = ast.get_source_segment(source, node)
            if not class_code:
                self.stats["skip_reasons"]["empty_class_code"] += 1
                continue

            # 优先使用 docstring
            docstring = ast.get_docstring(node)
            if docstring:
                instruction = docstring
            else:
                # 从类名生成指令
                class_name = node.name.replace("_", " ")
                methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                if methods:
                    method_list = "、".join(methods[:5])
                    if len(methods) > 5:
                        method_list += f" 等 {len(methods)} 个方法"
                    instruction = f"定义类 {class_name}，包含方法 {method_list}"
                else:
                    instruction = f"定义类 {class_name}"

            instruction = re.sub(r"\s+", " ", instruction).strip()

            samples.append({
                "type": "code_comment",
                "instruction": instruction,
                "code": class_code.strip(),
            })

        return samples

    def _extract_full_file_sample(self, source: str, file_path: Path) -> Optional[Dict[str, str]]:
        """
        提取完整文件作为训练样本

        新增功能：让模型学习完整的模块级代码
        """
        # 跳过太小的文件（可能不完整）
        if len(source.strip()) < 200:
            self.stats["skip_reasons"]["file_too_small"] += 1
            return None

        # 跳过太大的文件（超出限制）
        if len(source) > MAX_SAMPLE_LENGTH * 2:
            self.stats["skip_reasons"]["file_too_large"] += 1
            return None

        # 尝试从文件 AST 提取 docstring 作为指令
        try:
            tree = ast.parse(source)
            docstring = ast.get_docstring(tree)
            if docstring:
                instruction = docstring
            else:
                # 从文件名生成指令
                file_name = file_path.stem.replace("_", " ")
                instruction = f"实现 {file_name} 模块的完整功能"
        except Exception:
            file_name = file_path.stem.replace("_", " ")
            instruction = f"实现 {file_name} 模块的功能"

        return {
            "type": "code_comment",
            "instruction": instruction,
            "code": source,
        }

    def _hash_code(self, code: str) -> str:
        """计算代码的 MD5 哈希用于去重"""
        return hashlib.md5(code.encode("utf-8")).hexdigest()

    def _json_length(self, sample: Dict[str, str]) -> int:
        """计算样本的 JSON 字符串长度"""
        return len(json.dumps(sample, ensure_ascii=False))

    def _truncate_sample(self, sample: Dict[str, str], file_path: Path) -> Dict[str, str]:
        """截断过长的样本"""
        if self._json_length(sample) <= MAX_SAMPLE_LENGTH:
            return sample

        self.stats["skip_reasons"]["truncated_too_long"] += 1
        self._log_error(f"[WARNING] 样本超过 {MAX_SAMPLE_LENGTH} 字符，已截断: {file_path}")

        result = dict(sample)
        code_field = "code"

        while self._json_length(result) > MAX_SAMPLE_LENGTH:
            value = result.get(code_field, "")
            if isinstance(value, str) and len(value) > 100:
                keep_length = max(100, int(len(value) * 0.85))
                result[code_field] = value[:keep_length].rstrip()
            else:
                break

        return result

    def _validate_sample(self, sample: Dict[str, str]) -> bool:
        """验证样本质量"""
        code = sample.get("code", "")

        # 检查代码是否为空或太短
        if not code or len(code.strip()) < 30:
            self.stats["skip_reasons"]["code_too_short"] += 1
            return False

        # 检查指令是否为空或太短
        instruction = sample.get("instruction", "").strip()
        if len(instruction) < 5:
            self.stats["skip_reasons"]["instruction_too_short"] += 1
            return False

        # 检查代码是否包含字面量 \n（表示转义换行符有问题）
        if "\\n" in code and not re.search(r'"""[\s\S]*?"""', code):
            # 如果代码中大量出现 \n 而不是真正的换行，标记为可疑
            pass

        return True

    def _write_sample(self, output_file, sample: Dict[str, str], file_path: Path) -> bool:
        """写入单个样本到输出文件"""
        sample = self._truncate_sample(sample, file_path)

        if not self._validate_sample(sample):
            return False

        code = sample.get("code", "")
        code_hash = self._hash_code(code)

        if code_hash in self.seen_hashes:
            self.stats["skip_reasons"]["duplicate"] += 1
            return False

        self.seen_hashes.add(code_hash)

        output_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
        self.stats["written_by_type"][sample["type"]] += 1
        return True

    def _build_samples_for_file(self, file_path: Path) -> List[Dict[str, str]]:
        """
        为单个文件构建所有样本

        改进点：
        1. 只生成 code_comment 类型
        2. 提取完整文件 + 单个函数 + 单个类
        3. 增加数据多样性
        """
        source = self._read_file(file_path)
        if not source:
            self.stats["skip_reasons"]["empty_or_unreadable_file"] += 1
            return []

        tree = self._parse_ast(source, file_path)
        if tree is None:
            return []

        samples = []

        try:
            # 1. 提取完整文件样本（如果文件足够大）
            full_file_sample = self._extract_full_file_sample(source, file_path)
            if full_file_sample:
                samples.append(full_file_sample)

            # 2. 提取函数定义样本
            function_samples = self._extract_function_samples(source, tree)
            # 限制每个文件最多提取 5 个函数样本（避免某个文件产生过多样本）
            if len(function_samples) > 5:
                function_samples = self.random.sample(function_samples, 5)
            samples.extend(function_samples)

            # 3. 提取类定义样本
            class_samples = self._extract_class_samples(source, tree)
            # 限制每个文件最多提取 3 个类样本
            if len(class_samples) > 3:
                class_samples = self.random.sample(class_samples, 3)
            samples.extend(class_samples)

        except Exception as exc:
            self.stats["skip_reasons"]["sample_build_failed"] += 1
            self._log_error(f"[ERROR] 构建样本失败 {file_path}: {exc}")

        return samples

    def run(self) -> None:
        """主运行方法"""
        self._load_manifest()

        py_files = sorted(CLEANED_DIR.rglob("*.py"))
        self.stats["total_files"] = len(py_files)

        print(f"[INFO] 开始处理 {len(py_files)} 个文件...")

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
                    print(f"[PROGRESS] 已处理 {index}/{len(py_files)} 个文件，"
                          f"已生成 {sum(self.stats['written_by_type'].values())} 个样本")

    def print_summary(self) -> None:
        """打印统计摘要"""
        total_generated = sum(self.stats["generated_by_type"].values())
        total_written = sum(self.stats["written_by_type"].values())

        print("\n" + "=" * 50)
        print("Sample Builder Summary")
        print("=" * 50)
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
        print("=" * 50 + "\n")


if __name__ == "__main__":
    files = list(CLEANED_DIR.rglob("*.py"))
    print(f"[INFO] 扫描到 {len(files)} 个 cleaned Python 文件")

    builder = SampleBuilder()
    builder.run()
    builder.print_summary()