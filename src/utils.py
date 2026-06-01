import yaml
from pathlib import Path
from typing import Optional


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_corpus(path: str, max_phrases: Optional[int] = None) -> list[str]:
    corpus_path = Path(path)
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus file not found: {path}\n"
            "Create a plain-text file with one phrase per line."
        )
    with open(corpus_path, "r", encoding="utf-8") as f:
        phrases = [line.strip() for line in f if line.strip()]
    if max_phrases is not None:
        phrases = phrases[:max_phrases]
    return phrases
