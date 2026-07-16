"""Central configuration for the data pipeline."""

import os

from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


# GitHub API
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_API_URL = "https://api.github.com"

# Request settings
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
REQUESTS_PER_MINUTE = 30
USER_AGENT = "Python-Data-Pipeline/1.0"

# Proxy settings
PROXY_ENABLED = os.getenv("PROXY_ENABLED", "false").lower() == "true"

# Multiple proxies can be separated by commas.
# Example:
# HTTP_PROXIES=http://127.0.0.1:7890,http://username:password@host:port
# HTTP_PROXIES = [
#     proxy.strip()
#     for proxy in os.getenv("HTTP_PROXIES", "").split(",")
#     if proxy.strip()
# ]
#
# for proxy in HTTP_PROXIES:
#     if not proxy.startswith("http://"):
#         raise ValueError(f"Proxy must use http:// format: {proxy}")

# Output paths
RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", "data/raw")
LOG_DIR = os.getenv("LOG_DIR", "logs")

# Crawler defaults
DEFAULT_PAGE_SIZE = 100
MAX_PAGES_PER_RUN = 10

# Model path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "deepseek-coder-1.3b-base"