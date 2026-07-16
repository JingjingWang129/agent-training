"""Clean Python source files collected under data/raw."""

import ast
import io
import json
import re
import tokenize
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config.settings import LOG_DIR, RAW_DATA_DIR


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / RAW_DATA_DIR
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
LOGS_DIR = PROJECT_ROOT / LOG_DIR

SKIPPED_LOG = LOGS_DIR / "skipped_files.log"
ERROR_LOG = LOGS_DIR / "cleaning_errors.log"
MANIFEST_PATH = CLEANED_DIR / "manifest.json"

IP_PATTERN = re.compile(
    r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
)
EMAIL_PATTERN = re.compile(
    r"\b[\w\.-]+@[\w\.-]+\.\w+\b"
)
AWS_KEY_PATTERN = re.compile(
    r"\bAKIA[A-Z0-9]{16}\b"
)
PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN.*?PRIVATE KEY-----.*?-----END.*?PRIVATE KEY-----",
    re.DOTALL,
)


class DataCleaner:
    """Validate, sanitize, format, and save Python source files."""

    def __init__(self) -> None:
        CLEANED_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        self.manifest: List[Dict] = []
        self.stats = Counter(
            {
                "total": 0,
                "passed": 0,
                "syntax_error": 0,
                "too_short": 0,
                "cleaning_error": 0,
            }
        )

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().isoformat(timespec="seconds")

    @staticmethod
    def _append_log(log_path: Path, message: str) -> None:
        try:
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(
                    f"[{DataCleaner._timestamp()}] {message}\n"
                )
        except OSError as exc:
            print(f"[ERROR] 无法写入日志 {log_path}: {exc}")

    def _log_skipped(self, file_path: Path, reason: str) -> None:
        relative_path = file_path.relative_to(RAW_DIR)
        self._append_log(
            SKIPPED_LOG,
            f"{relative_path} | {reason}",
        )

    def _log_error(self, file_path: Path, error: Exception) -> None:
        try:
            relative_path = file_path.relative_to(RAW_DIR)
        except ValueError:
            relative_path = file_path

        self._append_log(
            ERROR_LOG,
            f"{relative_path} | {type(error).__name__}: {error}",
        )

    @staticmethod
    def _read_source(file_path: Path) -> str:
        """Read Python source while respecting its encoding declaration."""
        with tokenize.open(file_path) as source_file:
            return source_file.read()

    @staticmethod
    def _count_code_lines(source: str) -> int:
        """Count non-empty lines containing code rather than only comments."""
        code_line_numbers = set()

        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        ignored_types = {
            tokenize.ENCODING,
            tokenize.ENDMARKER,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.NEWLINE,
            tokenize.NL,
            tokenize.COMMENT,
        }

        for token in tokens:
            if token.type not in ignored_types and token.string.strip():
                code_line_numbers.add(token.start[0])

        return len(code_line_numbers)

    @staticmethod
    def _remove_pii(source: str) -> str:
        """Replace configured sensitive-data patterns."""
        source = PRIVATE_KEY_PATTERN.sub(
            "<REDACTED_PRIVATE_KEY>",
            source,
        )
        source = AWS_KEY_PATTERN.sub(
            "<REDACTED_AWS_KEY>",
            source,
        )
        source = EMAIL_PATTERN.sub(
            "<REDACTED_EMAIL>",
            source,
        )
        source = IP_PATTERN.sub(
            "<REDACTED_IP>",
            source,
        )
        return source

    @staticmethod
    def _normalize_comments(source: str) -> str:
        """Ensure ordinary comments have one space after the hash."""
        output_tokens = []

        tokens = tokenize.generate_tokens(io.StringIO(source).readline)

        for token in tokens:
            if token.type != tokenize.COMMENT:
                output_tokens.append(token)
                continue

            comment = token.string

            # Preserve shebang and source-encoding declarations.
            is_shebang = token.start[0] == 1 and comment.startswith("#!")
            is_encoding = bool(
                re.match(r"#.*coding[:=]\s*[-\w.]+", comment)
            )

            if is_shebang or is_encoding:
                output_tokens.append(token)
                continue

            comment_body = comment[1:].strip()
            normalized = "# " + comment_body if comment_body else "#"

            output_tokens.append(
                tokenize.TokenInfo(
                    token.type,
                    normalized,
                    token.start,
                    token.end,
                    token.line,
                )
            )

        return tokenize.untokenize(output_tokens)

    @staticmethod
    def _normalize_indentation(source: str) -> str:
        """Convert indentation tabs to spaces and trim trailing whitespace."""
        normalized_lines = []

        for line in source.splitlines():
            expanded_line = line.expandtabs(4)
            normalized_lines.append(expanded_line.rstrip())

        result = "\n".join(normalized_lines)
        return result.rstrip() + "\n"

    def _format_source(self, source: str) -> str:
        source = self._normalize_comments(source)
        source = self._normalize_indentation(source)
        return source

    @staticmethod
    def _extract_metadata(
        syntax_tree: ast.AST,
    ) -> Tuple[int, List[str]]:
        """Extract function count and top-level import names."""
        function_count = 0
        imports = set()

        for node in ast.walk(syntax_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_count += 1

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])

        return function_count, sorted(imports)

    def clean_file(self, file_path: Path) -> Optional[Dict]:
        """Clean one Python file and return its manifest entry."""
        try:
            original_size = file_path.stat().st_size
            source = self._read_source(file_path)

            try:
                ast.parse(source, filename=str(file_path))
            except SyntaxError as exc:
                self.stats["syntax_error"] += 1
                self._log_skipped(
                    file_path,
                    f"AST语法错误: 第 {exc.lineno} 行，{exc.msg}",
                )
                return None

            original_line_count = self._count_code_lines(source)

            if original_line_count < 10:
                self.stats["too_short"] += 1
                self._log_skipped(
                    file_path,
                    f"有效代码行数不足: {original_line_count} < 10",
                )
                return None

            cleaned_source = self._remove_pii(source)
            cleaned_source = self._format_source(cleaned_source)

            # Ensure sanitization and formatting did not invalidate the code.
            cleaned_tree = ast.parse(
                cleaned_source,
                filename=str(file_path),
            )

            cleaned_line_count = self._count_code_lines(cleaned_source)
            function_count, imports = self._extract_metadata(cleaned_tree)

            relative_path = file_path.relative_to(RAW_DIR)
            output_path = CLEANED_DIR / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(cleaned_source, encoding="utf-8")

            cleaned_size = len(cleaned_source.encode("utf-8"))

            manifest_entry = {
                "file_path": relative_path.as_posix(),
                "original_size": original_size,
                "cleaned_size": cleaned_size,
                "line_count": cleaned_line_count,
                "function_count": function_count,
                "imports": imports,
                "cleaned_at": self._timestamp(),
            }

            self.stats["passed"] += 1
            return manifest_entry

        except Exception as exc:
            self.stats["cleaning_error"] += 1
            self._log_error(file_path, exc)
            print(f"[ERROR] 清洗 {file_path} 失败: {exc}")
            return None

    def save_manifest(self) -> None:
        """Save metadata for all successfully cleaned files."""
        try:
            with MANIFEST_PATH.open("w", encoding="utf-8") as manifest_file:
                json.dump(
                    self.manifest,
                    manifest_file,
                    ensure_ascii=False,
                    indent=2,
                )
        except OSError as exc:
            self._log_error(MANIFEST_PATH, exc)
            print(f"[ERROR] 保存 manifest.json 失败: {exc}")

    def run(self) -> Counter:
        """Scan and clean all Python files under data/raw."""
        python_files = sorted(RAW_DIR.rglob("*.py"))
        self.stats["total"] = len(python_files)

        for file_path in python_files:
            manifest_entry = self.clean_file(file_path)

            if manifest_entry is not None:
                self.manifest.append(manifest_entry)

        self.save_manifest()
        return self.stats

    def print_summary(self) -> None:
        skipped = (
            self.stats["syntax_error"]
            + self.stats["too_short"]
            + self.stats["cleaning_error"]
        )

        print("\n========== 数据清洗统计 ==========")
        print(f"总文件数:       {self.stats['total']}")
        print(f"通过数:         {self.stats['passed']}")
        print(f"跳过数:         {skipped}")
        print(f"  AST 语法错误: {self.stats['syntax_error']}")
        print(f"  代码行数不足: {self.stats['too_short']}")
        print(f"  清洗异常:     {self.stats['cleaning_error']}")
        print(f"Manifest:       {MANIFEST_PATH}")
        print("==================================")


if __name__ == "__main__":
    raw_file_count = (
        len(list(RAW_DIR.rglob("*.py")))
        if RAW_DIR.exists()
        else 0
    )

    print(f"扫描目录: {RAW_DIR}")
    print(f"发现 {raw_file_count} 个 .py 文件")

    cleaner = DataCleaner()
    cleaner.run()
    cleaner.print_summary()