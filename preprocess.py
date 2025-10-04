import json
import re
from pathlib import Path

DATA_DIR = Path("data")

def load_json(filename):
    """Load a JSON file and return list of records."""
    with open(DATA_DIR / filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    return data if isinstance(data, list) else []

def extract_text(item):
    """Extract textual fields from a JSON record."""
    texts = []
    for k in ("text", "description", "body", "content", "name"):
        if k in item and item[k]:
            texts.append(str(item[k]))
    if "content_json" in item and isinstance(item["content_json"], dict):
        for v in item["content_json"].values():
            if isinstance(v, str) and v.strip():
                texts.append(v)
    return texts

def chunk_text(text, max_words=80):
    """Split long text into smaller chunks."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, cur, count = [], [], 0
    for s in sentences:
        words = s.split()
        if len(words) < 5:
            continue
        if count + len(words) > max_words and cur:
            chunks.append(" ".join(cur))
            cur, count = [s], len(words)
        else:
            cur.append(s)
            count += len(words)
    if cur:
        chunks.append(" ".join(cur))
    return chunks
