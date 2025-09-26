import json, re
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------
# Step 1: Load JSON data
# -------------------
def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    return data if isinstance(data, list) else []

content = load_json("Toolkit_Content_results.json")
resources = load_json("Toolkit_Resources_results.json")

# -------------------
# Step 2: Extract + chunk text
# -------------------
def extract_text(item):
    texts = []
    if isinstance(item, dict):
        for k in ("text", "description", "body", "content", "name"):
            if k in item and item[k]:
                texts.append(str(item[k]))
        if "content_json" in item and isinstance(item["content_json"], dict):
            for k,v in item["content_json"].items():
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

docs = []
for item in content + resources:
    for t in extract_text(item):
        docs.extend(chunk_text(t))

print(f"✅ Loaded {len(docs)} chunks from Toolkit JSONs")

# -------------------
# Step 3: Embed chunks
# -------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)

# -------------------
# Step 4: Retrieval function
# -------------------
def retrieve(query, top_k=3):
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores = (embeddings @ q_emb.T).squeeze()
    idxs = np.argsort(-scores)[:top_k]
    return [(float(scores[i]), docs[i]) for i in idxs]

# -------------------
# Step 5: Simple CLI
# -------------------
print("\nReady! Type a question (or 'exit' to quit).")
while True:
    q = input("\nQ> ").strip()
    if q.lower() in ("exit","quit"): break
    results = retrieve(q)
    print("\nTop matches:")
    for score, passage in results:
        print(f" - (score={score:.3f}) {passage[:200]}...")

