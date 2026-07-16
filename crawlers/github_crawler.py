"""GitHub Python repository crawler."""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from github import Auth, Github, GithubException
from github.Repository import Repository

from config.settings import GITHUB_TOKEN

class GitHubCrawler:
    def __init__(self, token: Optional[str] = None):
        self.token = GITHUB_TOKEN
        if not self.token:
            raise ValueError("请先设置 GITHUB_TOKEN 环境变量，或在初始化时传入 token")

        auth = Auth.Token(self.token)
        self.github = Github(auth=auth, timeout=30)

        self.project_root = Path(__file__).resolve().parents[1]
        self.raw_data_dir = self.project_root / "data" / "raw"
        self.metadata_path = self.project_root / "data" / "metadata.json"

        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        self.metadata = self._load_metadata()
        self._current_repo_updated_at: Optional[str] = None
        self._fetch_complete = False

    def _load_metadata(self) -> Dict[str, str]:
        if not self.metadata_path.exists():
            return {}

        try:
            with self.metadata_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            print(f"[ERROR] 读取 metadata.json 失败: {exc}")
            return {}

    def _save_metadata(self) -> None:
        try:
            with self.metadata_path.open("w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"[ERROR] 保存 metadata.json 失败: {exc}")

    def _wait_for_rate_limit(self) -> None:
        try:
            remaining, _ = self.github.rate_limiting
            reset_time = self.github.rate_limiting_resettime

            if remaining < 10:
                sleep_seconds = max(0, reset_time - int(time.time())) + 1
                reset_at = datetime.fromtimestamp(reset_time).strftime("%Y-%m-%d %H:%M:%S")
                print(f"[INFO] GitHub API 剩余请求次数不足，等待到 {reset_at} 后继续")
                time.sleep(sleep_seconds)
        except Exception as exc:
            print(f"[ERROR] 检查限流状态失败: {exc}")

    def _request_with_retry(self, func: Callable[[], Any], error_context: str = "") -> Any:
        last_error = None

        for attempt in range(3):
            try:
                self._wait_for_rate_limit()
                return func()
            except Exception as exc:
                last_error = exc
                wait_seconds = 2**attempt
                print(f"[ERROR] {error_context} 失败，第 {attempt + 1}/3 次重试: {exc}")
                time.sleep(wait_seconds)

        raise last_error

    def search_repos(
        self,
        query: str = "language:python stars:>50",
        max_results: int = 50,
    ) -> List[Repository]:
        repos: List[Repository] = []

        if max_results <= 0:
            return repos

        try:
            paginated_repos = self._request_with_retry(
                lambda: self.github.search_repositories(
                    query=query,
                    sort="stars",
                    order="desc",
                ),
                "搜索仓库",
            )

            page_index = 0

            while len(repos) < max_results:
                current_page = page_index  # 创建局部变量副本
                page = self._request_with_retry(
                    lambda: paginated_repos.get_page(current_page),
                    f"获取搜索结果第 {current_page + 1} 页",
                )

                if not page:
                    break

                for repo in page:
                    repos.append(repo)
                    if len(repos) >= max_results:
                        break

                page_index += 1

        except Exception as exc:
            print(f"[ERROR] 搜索仓库失败: {exc}")

        return repos

    def _should_skip_repo(self, repo_full_name: str, updated_at: datetime) -> bool:
        latest_updated_at = updated_at.isoformat()
        old_updated_at = self.metadata.get(repo_full_name)

        if old_updated_at == latest_updated_at:
            print(f"[INFO] 仓库未更新，跳过: {repo_full_name}")
            return True

        self._current_repo_updated_at = latest_updated_at
        return False

    def fetch_repo_contents(self, repo_full_name: str) -> Dict[str, str]:
        contents_dict: Dict[str, str] = {}
        self._fetch_complete = False
        self._current_repo_updated_at = None

        try:
            repo = self._request_with_retry(
                lambda: self.github.get_repo(repo_full_name),
                f"获取仓库 {repo_full_name}",
            )

            if self._should_skip_repo(repo_full_name, repo.updated_at):
                self._fetch_complete = True
                return contents_dict

            queue = self._request_with_retry(
                lambda: repo.get_contents(""),
                f"读取仓库根目录 {repo_full_name}",
            )

            if not isinstance(queue, list):
                queue = [queue]

            while queue:
                item = queue.pop(0)

                try:
                    if item.type == "dir":
                        current_item = item  # 创建局部变量副本
                        children = self._request_with_retry(
                            lambda: repo.get_contents(current_item.path),
                            f"读取目录 {repo_full_name}/{current_item.path}",
                        )

                        if isinstance(children, list):
                            queue.extend(children)
                        else:
                            queue.append(children)

                    elif item.type == "file" and item.path.endswith(".py"):
                        current_item = item  # 创建局部变量副本
                        file_content = self._request_with_retry(
                            lambda: repo.get_contents(current_item.path),
                            f"读取文件 {repo_full_name}/{current_item.path}",
                        )

                        if file_content.encoding == "base64":
                            decoded = file_content.decoded_content.decode(
                                "utf-8",
                                errors="replace",
                            )
                            contents_dict[item.path] = decoded

                except Exception as exc:
                    print(f"[ERROR] 处理 {repo_full_name}/{getattr(item, 'path', '')} 失败: {exc}")
                    continue

            self._fetch_complete = True

        except GithubException as exc:
            print(f"[ERROR] 获取仓库 {repo_full_name} 失败: {exc}")
        except Exception as exc:
            print(f"[ERROR] 获取仓库 {repo_full_name} 失败: {exc}")

        return contents_dict

    def save_raw_data(self, repo_full_name: str, contents_dict: Dict[str, str]) -> None:
        repo_name = repo_full_name.split("/")[-1]
        repo_dir = self.raw_data_dir / repo_name
        repo_dir.mkdir(parents=True, exist_ok=True)

        saved_count = 0

        for file_path, content in contents_dict.items():
            try:
                output_path = repo_dir / file_path
                resolved_output_path = output_path.resolve()
                resolved_repo_dir = repo_dir.resolve()

                if not str(resolved_output_path).startswith(str(resolved_repo_dir)):
                    print(f"[ERROR] 非法文件路径，已跳过: {file_path}")
                    continue

                output_path.parent.mkdir(parents=True, exist_ok=True)

                with output_path.open("w", encoding="utf-8") as f:
                    f.write(content)

                saved_count += 1

            except Exception as exc:
                print(f"[ERROR] 保存文件 {file_path} 失败: {exc}")
                continue

        if self._fetch_complete and self._current_repo_updated_at:
            self.metadata[repo_full_name] = self._current_repo_updated_at
            self._save_metadata()

        print(f"[SUCCESS] 已抓取: {repo_name}，共 {saved_count} 个 .py 文件")


if __name__ == "__main__":
    crawler = GitHubCrawler()

    # 抓取 50 个高质量仓库
    repos = crawler.search_repos(
        query="language:python stars:>100 forks:>10 pushed:>2025-01-01",
        max_results=50
    )

    for repo in repos:
        full_name = repo.full_name
        contents = crawler.fetch_repo_contents(full_name)
        if contents:
            crawler.save_raw_data(full_name, contents)