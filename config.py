from pathlib import Path

# Repo root is one level up from this file
REPO_DIR = Path(__file__).resolve().parents[0]

DATA_DIR = REPO_DIR / "data" / "fiqa"
INDEX_DIR = REPO_DIR / "indexes"

# Model configurations
MODEL_CONFIGS = {
    "gpt-4.1-nano": {
        "max_context": 1_047_576,
        "max_output": 32_768,
        "encoding": "o200k_base",
        "documentation": "https://developers.openai.com/api/docs/models/gpt-4.1-nano",
    },
    "gpt-5-nano": {
        "max_context": 400_000,
        "max_output": 128_000,
        "encoding": "o200k_base",
        "documentation": "https://developers.openai.com/api/docs/models/gpt-5-nano",
    },
}
