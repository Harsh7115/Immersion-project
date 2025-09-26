import json, re, pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

DATA_DIR = Path("data")

def load_json(filename):
    with open(DATA_DIR/filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    return data if isinstance(data, list) else []

def extract_text(item):
    texts = []
    if isinstance(item, dict):
        for k in ("text", "description", "body", "content", "name"):
            if k in item and item[k]:
                texts.append(str(item[k]))
        if "content_json" in item and isinstance(item["content_json"], dict):
            for v in item["content_json"].values():
                if isinstance(v, str) and v.strip():
                    texts.append(v)
    return texts

def chunk_text(text, max_words=80):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, cur, count = [], [], 0
    for s in sentences:
        words = s.split()
        if len(words) < 5: continue
        if count + len(words) > max_words and cur:
            chunks.append(" ".join(cur))
            cur, count = [s], len(words)
        else:
            cur.append(s); count += len(words)
    if cur: chunks.append(" ".join(cur))
    return chunks

print("🔄 Loading JSON...")
content = load_json("Toolkit_Content_results.json")
resources = load_json("Toolkit_Resources_results.json")

docs = []
for item in content + resources:
    for t in extract_text(item):
        docs.extend(chunk_text(t))
print(f"✅ Loaded {len(docs)} chunks")

print("🔄 Encoding with SentenceTransformer...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)

# Save
print("💾 Saving artifacts...")
np.save(DATA_DIR/"embeddings.npy", embeddings)
with open(DATA_DIR/"docs.pkl", "wb") as f:
    pickle.dump(docs, f)
print("✅ Done!")
