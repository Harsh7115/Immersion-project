import json, re, pickle
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
EMB_FILE = DATA_DIR / "embeddings.npy"
CHUNK_FILE = DATA_DIR / "chunks.pkl"        # (chunks, chunk_to_doc_idx)
DOC_FILE = DATA_DIR / "documents.pkl"       # list of full docs
META_FILE = DATA_DIR / "metadata.pkl"       # metadata for each doc
CONTENT_FILE = DATA_DIR / "Toolkit_Content_results.json"
RESOURCES_FILE = DATA_DIR / "Toolkit_Resources_results.json"

model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- Helpers ----------
def _load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    return data if isinstance(data, list) else []

def _extract_text(item):
    texts = []
    for k in ("text", "description", "body", "content", "name"):
        if k in item and item[k]:
            texts.append(str(item[k]))
    if "content_json" in item and isinstance(item["content_json"], dict):
        for v in item["content_json"].values():
            if isinstance(v, str) and v.strip():
                texts.append(v)
    return texts

def _chunk_text(text, max_words=80):
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

def _build_index():
    print("🔄 Building index...")
    content = _load_json(CONTENT_FILE)
    resources = _load_json(RESOURCES_FILE)

    chunks = []
    chunk_to_doc_idx = []
    documents = []
    metadata = []   # will store dict with name, id, dates

    for item in content + resources:
        # Combine all text for embeddings
        full_text = "\n".join(_extract_text(item))
        if not full_text.strip():
            continue

        doc_idx = len(documents)
        documents.append(full_text)

        # --- Metadata ---
        meta = {
            "document_id": item.get("document_id"),
            "name": item.get("name"),
            "create_date": item.get("create_date"),
            "publish_date": item.get("publish_date"),
            "categories": item.get("categories")
        }
        metadata.append(meta)

        # --- Chunking ---
        for ch in _chunk_text(full_text):
            chunks.append(ch)
            chunk_to_doc_idx.append(doc_idx)

    print(f"✅ Created {len(chunks)} chunks from {len(documents)} documents")
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

    np.save(EMB_FILE, embeddings)
    with open(CHUNK_FILE, "wb") as f:
        pickle.dump((chunks, chunk_to_doc_idx), f)
    with open(DOC_FILE, "wb") as f:
        pickle.dump(documents, f)
    with open(META_FILE, "wb") as f:
        pickle.dump(metadata, f)
    print("✅ Index saved!")

def _load_or_build():
    if not (EMB_FILE.exists() and CHUNK_FILE.exists() and DOC_FILE.exists() and META_FILE.exists()):
        _build_index()
    print("🔄 Loading precomputed data...")
    embeddings = np.load(EMB_FILE)
    with open(CHUNK_FILE, "rb") as f:
        chunks, chunk_to_doc_idx = pickle.load(f)
    with open(DOC_FILE, "rb") as f:
        documents = pickle.load(f)
    with open(META_FILE, "rb") as f:
        metadata = pickle.load(f)
    return chunks, chunk_to_doc_idx, documents, metadata, embeddings

chunks, chunk_to_doc_idx, documents, metadata, embeddings = _load_or_build()

# ---------- Retrieval ----------
def retrieve(query, top_k=5):
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores = (embeddings @ q_emb.T).squeeze()

    # Aggregate: pick the max scoring chunk per document
    doc_best = defaultdict(lambda: (-np.inf, None))  # (score, best_snippet)
    for idx, sc in enumerate(scores):
        doc_id = chunk_to_doc_idx[idx]
        if sc > doc_best[doc_id][0]:
            doc_best[doc_id] = (sc, chunks[idx])

    ranked = sorted(doc_best.items(), key=lambda x: -x[1][0])[:top_k]
    results = []
    for doc_id, (score, snippet) in ranked:
        results.append({
            "score": float(score),
            "snippet": snippet,
            "full_text": documents[doc_id],
            "metadata": metadata[doc_id]
        })
    return results
