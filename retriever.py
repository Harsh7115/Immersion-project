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
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", use_auth_token=False)
models = {
    "bge": SentenceTransformer("BAAI/bge-base-en-v1.5"),
    "multiqa": SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
}

# --------- Build Index ---------
def build(model_name=None):
    print("🔄 Building index...")

    # Load all source data
    content = load_json("Toolkit_Content_results.json")
    # resources = load_json("Toolkit_Resources_results.json")
    # all_items = content + resources
    all_items = content

    if not (DOC_FILE.exists() and META_FILE.exists()):
        print("Metadata missing, Building shared documents and metadata...")
        content = load_json("Toolkit_Content_results.json")
        all_items = content  # you commented out resources; keep flexible

        documents, metadata = [], []
        for item in all_items:
            full_text = "\n".join(extract_text(item)).strip()
            if not full_text:
                continue
            documents.append(full_text)
            metadata.append({
                "document_id": item.get("document_id"),
                "name": item.get("name"),
                "create_date": item.get("create_date"),
                "publish_date": item.get("publish_date"),
                "categories": item.get("categories"),
            })

        with open(DOC_FILE, "wb") as f:
            pickle.dump(documents, f)
        with open(META_FILE, "wb") as f:
            pickle.dump(metadata, f)
        print(f"Saved {len(documents)} documents and metadata.")
    else:
        with open(DOC_FILE, "rb") as f:
            documents = pickle.load(f)
        with open(META_FILE, "rb") as f:
            metadata = pickle.load(f)

    # For each embedding model
    mdl = models.get(model_name, None)
    if mdl:
        print(f"\nProcessing model: {model_name}")
        chunks, chunk_to_doc_idx = [], []
        for doc_idx, text in enumerate(documents):
            doc_chunks = chunk_text(text, model=mdl)
            for ch in doc_chunks:
                chunks.append(ch)
                chunk_to_doc_idx.append(doc_idx)
        print(f"  → {len(chunks)} chunks created for {model_name}")

        # Encode with the same model
        embeddings = mdl.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

        # Save per model
        np.save(DATA_DIR / f"embeddings_{model_name}.npy", embeddings)
        with open(DATA_DIR / f"chunks_{model_name}.pkl", "wb") as f:
            pickle.dump((chunks, chunk_to_doc_idx), f)
        print(f"Saved embeddings_{model_name}.npy and chunks_{model_name}.pkl")

def load():
    missing = [
        n for n in models.keys()
        if not (DATA_DIR / f"embeddings_{n}.npy").exists()
    ]
    for modelName in missing:
        print(f"Missing embeddings for: {modelName}, rebuilding index...")
        build(modelName)
    print("Loading Embeddings...")
    embeddings = {}
    chunks_per_model = {}
    chunk_to_doc_idx = None  # shared structure (per model identical order)

    for name in models.keys():
        embeddings[name] = np.load(DATA_DIR / f"embeddings_{name}.npy", allow_pickle=True)
        with open(DATA_DIR / f"chunks_{name}.pkl", "rb") as f:
            chunks, chunk_to_doc_idx = pickle.load(f)
        chunks_per_model[name] = chunks

    with open(DOC_FILE, "rb") as f:
        documents = pickle.load(f)
    with open(META_FILE, "rb") as f:
        metadata = pickle.load(f)

    load_custom_vocab(documents)
    return chunks_per_model, chunk_to_doc_idx, documents, metadata, embeddings




chunks_per_model, chunk_to_doc_idx, documents, metadata, embeddings = load()

# --------- Retrieval ---------
def retrieve(query, top_k=5, weights=None):
    weights = weights or {"bge": 0.5, "multiqa": 0.5}

    # --- Step 1: Spell correction ---
    query, suggestion = autocorrect_query(query)
    if suggestion:
        print(f"💡 Did you mean: {suggestion}?")

    # --- Step 2: Aggregate doc-level scores ---
    doc_scores = defaultdict(float)
    doc_counts = defaultdict(float)

    for name, mdl in models.items():
        if name not in embeddings:
            print(f"⚠️ Skipping model '{name}' (embeddings missing)")
            continue

        w = weights.get(name, 1.0)
        q_emb = mdl.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores = (embeddings[name] @ q_emb.T).squeeze()
        chunks = chunks_per_model[name]

        # Load corresponding mapping
        with open(DATA_DIR / f"chunks_{name}.pkl", "rb") as f:
            _, chunk_to_doc_idx = pickle.load(f)

        # Aggregate max score per document
        doc_best = defaultdict(lambda: -np.inf)
        for idx, sc in enumerate(scores):
            doc_id = chunk_to_doc_idx[idx]
            if sc > doc_best[doc_id]:
                doc_best[doc_id] = sc

        # Add to combined doc scores (weighted)
        for doc_id, sc in doc_best.items():
            doc_scores[doc_id] += w * sc
            doc_counts[doc_id] += w

    # --- Step 3: Normalize weighted scores ---
    for doc_id in doc_scores:
        doc_scores[doc_id] /= doc_counts[doc_id]

    # --- Step 4: Rank documents by score ---
    ranked = sorted(doc_scores.items(), key=lambda x: -x[1])[:top_k]

    # --- Step 5: Select representative snippet ---
    results = []
    for doc_id, score in ranked:
        # use BGE chunks as default snippet source (or fallback)
        base_chunks = chunks_per_model.get("bge") or next(iter(chunks_per_model.values()))
        # pick best-scoring snippet for that doc
        best_snippet = ""
        best_score = -np.inf
        for name in models.keys():
            with open(DATA_DIR / f"chunks_{name}.pkl", "rb") as f:
                chunks, chunk_to_doc_idx = pickle.load(f)
            q_emb = models[name].encode([query], convert_to_numpy=True, normalize_embeddings=True)
            scores = (embeddings[name] @ q_emb.T).squeeze()
            for idx, sc in enumerate(scores):
                if chunk_to_doc_idx[idx] == doc_id and sc > best_score:
                    best_score = sc
                    best_snippet = chunks[idx]
        results.append({
            "score": float(score),
            "snippet": best_snippet.strip(),
            "full_text": documents[doc_id],
            "metadata": metadata[doc_id],
        })

    return results
