"""
submission/corpus_utils.py

Small, shared, non-graded utility for reading the corpus format used
throughout this assignment. You are welcome to use this as-is — reading a
JSONL file is plumbing, not the ranking logic the assignment grades — or
replace it if your index needs a different loading strategy (e.g.
streaming instead of loading everything into memory).
"""
import json
from typing import Iterator, List, Tuple


def iter_corpus(path: str) -> Iterator[Tuple[str, str]]:
    """Yield corpus records one at a time without retaining document text."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield obj["doc_id"], obj["text"]


def load_corpus(path: str) -> List[Tuple[str, str]]:
    """Read a corpus.jsonl file (one {"doc_id": ..., "text": ...} object
    per line) and return a list of (doc_id, text) pairs in file order."""
    return list(iter_corpus(path))
