import json, re, pickle, os
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from huggingface_hub import InferenceClient, login
from symspellpy import SymSpell, Verbosity

from preprocess import load_json, extract_text, chunk_text
from spellcheck import autocorrect_query, load_custom_vocab

# To be removed # Initialize SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary("data/freqdict.txt", term_index=0, count_index=1)

def spell_correct(query: str) -> str:
    suggestions = sym_spell.lookup(query, Verbosity.CLOSEST, max_edit_distance=2)
    if suggestions:
        return suggestions[0].term
    return query

# --------- Paths ---------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
EMB_FILE = DATA_DIR / "embeddings.npy"
CHUNK_FILE = DATA_DIR / "chunks.pkl"        # (chunks, chunk_to_doc_idx)
DOC_FILE = DATA_DIR / "documents.pkl"       # list of full docs
META_FILE = DATA_DIR / "metadata.pkl"       # metadata for each doc
CONTENT_FILE = DATA_DIR / "Toolkit_Content_results.json"
RESOURCES_FILE = DATA_DIR / "Toolkit_Resources_results.json"

# os.environ.pop("HF_HUB_TOKEN", None)
# os.environ.pop("HUGGINGFACE_TOKEN", None)
# os.environ.pop("HF_TOKEN", None)
# login(token=None, add_to_git_credential=False)

# Embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", use_auth_token=False)

# --------- Build Index ---------
def _build_index():
    print("🔄 Building index...")
    content = load_json("Toolkit_Content_results.json")
    resources = load_json("Toolkit_Resources_results.json")

    chunks, chunk_to_doc_idx, documents, metadata = [], [], [], []

    for item in content + resources:
        full_text = "\n".join(extract_text(item))
        if not full_text.strip():
            continue

        doc_idx = len(documents)
        documents.append(full_text)

        meta = {
            "document_id": item.get("document_id"),
            "name": item.get("name"),
            "create_date": item.get("create_date"),
            "publish_date": item.get("publish_date"),
            "categories": item.get("categories"),
        }
        metadata.append(meta)

        for ch in chunk_text(full_text):
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
    embeddings = np.load(EMB_FILE, allow_pickle=True)
    with open(CHUNK_FILE, "rb") as f:
        chunks, chunk_to_doc_idx = pickle.load(f)
    with open(DOC_FILE, "rb") as f:
        documents = pickle.load(f)
    with open(META_FILE, "rb") as f:
        metadata = pickle.load(f)
    load_custom_vocab(documents)
    return chunks, chunk_to_doc_idx, documents, metadata, embeddings

# Load on import
chunks, chunk_to_doc_idx, documents, metadata, embeddings = _load_or_build()

# --------- Retrieval ---------
def retrieve(query, top_k=5):
    # Autocorrect step
    query, suggestion = autocorrect_query(query)
    if suggestion:
        print(suggestion)  # Logs correction suggestion in console

    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores = (embeddings @ q_emb.T).squeeze()

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
