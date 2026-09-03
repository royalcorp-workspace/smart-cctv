"""Configuration loader with environment variable expansion and URL encoding."""

import json
import os
import re
import urllib.parse
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv

# Load .env from root workspace
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH)
else:
    load_dotenv()


def _expand_raw_env(s: str) -> str:
    """Expand ${VAR_NAME}, $VAR_NAME, and %VAR_NAME% into raw string values."""
    res = re.sub(
        r"\$\{([A-Za-z0-9_]+)\}",
        lambda m: os.environ.get(m.group(1), m.group(0)),
        s,
    )
    res = re.sub(
        r"(?<!\\)\$([A-Za-z0-9_]+)",
        lambda m: os.environ.get(m.group(1), m.group(0)),
        res,
    )
    res = os.path.expandvars(res)
    return res


def expand_env_string(val: str) -> str:
    """Expand environment variables in strings.

    If the string is an RTSP or HTTP URL with credentials (protocol://user:pass@host),
    automatically URL-encodes user and password with urllib.parse.quote(..., safe='')
    so that special characters like '#' (%23) do not corrupt URL parsing in FFmpeg/OpenCV.
    """
    # Check if string matches a URL with userinfo: scheme://userinfo@host:port/path
    url_match = re.match(r"^([a-zA-Z0-9_+.-]+://)([^@]+)@(.+)$", val)
    if url_match:
        scheme_prefix = url_match.group(1)
        userinfo_template = url_match.group(2)
        host_and_path = url_match.group(3)

        if ":" in userinfo_template:
            raw_user, raw_pass = userinfo_template.split(":", 1)
            expanded_user = _expand_raw_env(raw_user)
            expanded_pass = _expand_raw_env(raw_pass)
            # URL-encode user and password with safe="" so special chars like '#' become '%23'
            encoded_user = urllib.parse.quote(expanded_user, safe="")
            encoded_pass = urllib.parse.quote(expanded_pass, safe="")
            encoded_userinfo = f"{encoded_user}:{encoded_pass}"
        else:
            expanded_user = _expand_raw_env(userinfo_template)
            encoded_userinfo = urllib.parse.quote(expanded_user, safe="")

        expanded_host_path = _expand_raw_env(host_and_path)
        return f"{scheme_prefix}{encoded_userinfo}@{expanded_host_path}"

    # Standard non-URL string expansion
    return _expand_raw_env(val)


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
    """Load JSON config file and expand environment variables with URL encoding."""
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return expand_env_vars(data)
