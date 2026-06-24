"""Tiny dependency-free .env loader.

Reads KEY=VALUE pairs from the project-root `.env`. Tolerates spaces around `=`
and surrounding single/double quotes, so both of these parse identically:
    TREXQUANT_API="abc123"
    TREXQUANT_API = "abc123"
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "..", ".env")


def load_env(path: str = ENV_PATH) -> dict:
    env = {}
    if not os.path.exists(path):
        return env
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key:
                env[key] = val
    return env


def get_token(key: str = "TREXQUANT_API", path: str = ENV_PATH):
    """Return the access token from the environment, preferring a real env var,
    then the .env file. Returns None if not found."""
    if os.environ.get(key):
        return os.environ[key]
    return load_env(path).get(key)
