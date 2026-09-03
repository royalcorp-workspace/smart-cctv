"""Configuration loader with environment variable expansion."""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv

# Load .env from root workspace
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH)
else:
    load_dotenv()


def expand_env_string(val: str) -> str:
    """Expand ${VAR_NAME}, $VAR_NAME, and %VAR_NAME% from environment variables."""
    # Match ${VAR_NAME}
    result = re.sub(
        r"\$\{([A-Za-z0-9_]+)\}",
        lambda m: os.environ.get(m.group(1), m.group(0)),
        val,
    )
    # Match $VAR_NAME
    result = re.sub(
        r"(?<!\\)\$([A-Za-z0-9_]+)",
        lambda m: os.environ.get(m.group(1), m.group(0)),
        result,
    )
    # Match Windows %VAR_NAME%
    result = os.path.expandvars(result)
    return result


def expand_env_vars(obj: Any) -> Any:
    """Recursively expand environment variables in dict, list, or string."""
    if isinstance(obj, str):
        expanded = expand_env_string(obj)
        if expanded.isdigit():
            return int(expanded)
        return expanded
    elif isinstance(obj, dict):
        return {k: expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [expand_env_vars(elem) for elem in obj]
    return obj


def load_camera_config(config_path: Path) -> Dict[str, Any]:
    """Load JSON config file and expand environment variables."""
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return expand_env_vars(data)
