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
CONTENT_FILE = DATA_DIR / "internalData.json"
RESOURCES_FILE = DATA_DIR / "externalData.json"

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
def build_internal(model_name=None):
    print(f"🔄 Building INTERNAL index using {model_name}...")

    mdl = models.get(model_name)
    if not mdl:
        print(f"⚠️ Model {model_name} not found in models dict.")
        return

    # Load internal Toolkit data
    content = load_json(str(CONTENT_FILE))
    all_items = content  # could include resources later

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
            "uri": item.get("uri")
        })

    # Save full docs + metadata
    with open(DOC_FILE, "wb") as f:
        pickle.dump(documents, f)
    with open(META_FILE, "wb") as f:
        pickle.dump(metadata, f)

    # Chunking
    chunks, chunk_to_doc_idx = [], []
    for doc_idx, text in enumerate(documents):
        for ch in chunk_text(text, model=mdl):
            chunks.append(ch)
            chunk_to_doc_idx.append(doc_idx)
    print(f"  → {len(chunks)} internal chunks created for {model_name}")

    embeddings = mdl.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

    # Save artifacts
    np.save(DATA_DIR / f"embeddings_{model_name}.npy", embeddings)
    with open(DATA_DIR / f"chunks_{model_name}.pkl", "wb") as f:
        pickle.dump((chunks, chunk_to_doc_idx), f)

    print(f"✅ Internal index built and saved for {model_name}")

def build_external(model_name=None):
    """Build index for external JSON documents."""
    print(f"🔄 Building EXTERNAL index using {model_name}...")

    EXTERNAL_JSON = DATA_DIR / "externalData.json"
    EXTERNAL_DIR = DATA_DIR / "external"
    EXTERNAL_DIR.mkdir(exist_ok=True)

    if not EXTERNAL_JSON.exists():
        raise FileNotFoundError(f"{EXTERNAL_JSON} not found")

    mdl = models.get(model_name)
    if not mdl:
        print(f"⚠️ Model {model_name} not found in models dict.")
        return

    with open(EXTERNAL_JSON, "r") as f:
        items = json.load(f)

    documents, metadata = [], []
    for entry in items:
        text = f"{entry.get('title', '')}\n\n{entry.get('content', '')}".strip()
        if not text:
            continue
        documents.append(text)
        metadata.append({
            "title": entry.get("title", ""),
            "url": entry.get("url", ""),
            "source_domain": entry.get("source_domain", "")
        })

    # Save docs + metadata
    with open(EXTERNAL_DIR / "documents.pkl", "wb") as f:
        pickle.dump(documents, f)
    with open(EXTERNAL_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    chunks, chunk_to_doc_idx = [], []
    for doc_idx, text in enumerate(documents):
        for ch in chunk_text(text, model=mdl):
            chunks.append(ch)
            chunk_to_doc_idx.append(doc_idx)
    print(f"  → {len(chunks)} external chunks created for {model_name}")

    embeddings = mdl.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    np.save(EXTERNAL_DIR / f"embeddings_{model_name}.npy", embeddings)
    with open(EXTERNAL_DIR / f"chunks_{model_name}.pkl", "wb") as f:
        pickle.dump((chunks, chunk_to_doc_idx), f)

    print(f"✅ External index built and saved for {model_name}")

def buildLoad(base_dir: Path, json_file: Path, prefix: str, builder_fn, models: dict):
    base_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "embeddings": {},
        "chunks": {},
        "chunk_to_doc_idx": None,
        "documents": [],
        "metadata": []
    }

    missing = [m for m in models.keys() if not (base_dir / f"embeddings_{m}.npy").exists()]
    if missing and json_file.exists():
        print(f"Missing {prefix.upper()} embeddings for {missing}, rebuilding...")
        for m in missing:
            builder_fn(m)
    elif not json_file.exists():
        print(f"{prefix.capitalize()} JSON file not found at {json_file}. Skipping.")
        return data

    for name in models.keys():
        emb_path = base_dir / f"embeddings_{name}.npy"
        chunk_path = base_dir / f"chunks_{name}.pkl"
        if not emb_path.exists() or not chunk_path.exists():
            continue
        data["embeddings"][name] = np.load(emb_path, allow_pickle=True)
        with open(chunk_path, "rb") as f:
            chunks, data["chunk_to_doc_idx"] = pickle.load(f)
        data["chunks"][name] = chunks

    docs_file = base_dir / "documents.pkl"
    meta_file = base_dir / "metadata.pkl"
    if docs_file.exists():
        with open(docs_file, "rb") as f:
            data["documents"] = pickle.load(f)
    if meta_file.exists():
        with open(meta_file, "rb") as f:
            data["metadata"] = pickle.load(f)

    return data

def load():
    INTERNAL_JSON = DATA_DIR / "internalData.json"
    EXTERNAL_JSON = DATA_DIR / "externalData.json"

    internal_data = buildLoad(
        base_dir=DATA_DIR,
        json_file=INTERNAL_JSON,
        prefix="internal",
        builder_fn=build_internal,
        models=models
    )

    external_data = buildLoad(
        base_dir=DATA_DIR/"external",
        json_file=EXTERNAL_JSON,
        prefix="external",
        builder_fn=build_external,
        models=models
    )

    # -------- Spell Correction Vocabulary --------
    all_docs = internal_data["documents"] + external_data["documents"]
    if all_docs:
        load_custom_vocab(all_docs)
        print("Custom vocabulary loaded for spell correction.\n")

    print("✅ All embeddings successfully loaded.\n")

    return {
        "internal": internal_data,
        "external": external_data
    }


retrieval_data = load()

# --------- Retrieval ---------
def retrieve_internal(query, top_k=5, weights=None):
    """Retrieve top internal documents for a query."""
    global retrieval_data
    weights = weights or {"bge": 0.5, "multiqa": 0.5}

    internal = retrieval_data["internal"]

    # --- Step 1: Spell correction ---
    query, suggestion = autocorrect_query(query)
    if suggestion:
        print(f"💡 Did you mean: {suggestion}?")

    embeddings = internal["embeddings"]
    chunks_dict = internal["chunks"]
    chunk_to_doc_idx = internal["chunk_to_doc_idx"]
    documents = internal["documents"]
    metadata = internal["metadata"]

    doc_scores = defaultdict(float)
    doc_counts = defaultdict(float)

    # --- Step 2: Compute hybrid scores ---
    for model_name, mdl in models.items():
        if model_name not in embeddings:
            continue

        w = weights.get(model_name, 1.0)
        q_emb = mdl.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores = (embeddings[model_name] @ q_emb.T).squeeze()

        doc_best = defaultdict(lambda: -np.inf)
        for idx, sc in enumerate(scores):
            doc_id = chunk_to_doc_idx[idx]
            if sc > doc_best[doc_id]:
                doc_best[doc_id] = sc

        for doc_id, sc in doc_best.items():
            doc_scores[doc_id] += w * sc
            doc_counts[doc_id] += w

    for doc_id in doc_scores:
        doc_scores[doc_id] /= doc_counts[doc_id]

    ranked = sorted(doc_scores.items(), key=lambda x: -x[1])[:top_k]

    # --- Step 3: Pick best snippets ---
    results = []
    for doc_id, score in ranked:
        best_snippet = ""
        best_score = -np.inf
        for model_name, mdl in models.items():
            if model_name not in embeddings:
                continue
            q_emb = mdl.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            scores = (embeddings[model_name] @ q_emb.T).squeeze()
            for idx, sc in enumerate(scores):
                if chunk_to_doc_idx[idx] == doc_id and sc > best_score:
                    best_score = sc
                    best_snippet = chunks_dict[model_name][idx]

        results.append({
            "score": float(score),
            "snippet": best_snippet.strip(),
            "full_text": documents[doc_id],
            "metadata": metadata[doc_id],
        })

    return results


def retrieve_external(query, top_k=5, weights=None):
    """Retrieve top external URLs for a query."""
    global retrieval_data
    weights = weights or {"bge": 0.5, "multiqa": 0.5}

    external = retrieval_data["external"]
    if not external["documents"]:
        print("⚠️ No external documents loaded.")
        return []

    # --- Step 1: Spell correction ---
    query, suggestion = autocorrect_query(query)
    if suggestion:
        print(f"💡 Did you mean: {suggestion}?")

    embeddings = external["embeddings"]
    chunks_dict = external["chunks"]
    chunk_to_doc_idx = external["chunk_to_doc_idx"]
    metadata = external["metadata"]

    doc_scores = defaultdict(float)
    doc_counts = defaultdict(float)

    # --- Step 2: Compute hybrid scores ---
    for model_name, mdl in models.items():
        if model_name not in embeddings:
            continue

        w = weights.get(model_name, 1.0)
        q_emb = mdl.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores = (embeddings[model_name] @ q_emb.T).squeeze()

        doc_best = defaultdict(lambda: -np.inf)
        for idx, sc in enumerate(scores):
            doc_id = chunk_to_doc_idx[idx]
            if sc > doc_best[doc_id]:
                doc_best[doc_id] = sc

        for doc_id, sc in doc_best.items():
            doc_scores[doc_id] += w * sc
            doc_counts[doc_id] += w

    for doc_id in doc_scores:
        doc_scores[doc_id] /= doc_counts[doc_id]

    ranked = sorted(doc_scores.items(), key=lambda x: -x[1])[:top_k]

    results = []
    for doc_id, score in ranked:
        meta = metadata[doc_id]
        results.append({
            "score": float(score),
            "title": meta.get("title", ""),
            "url": meta.get("url", ""),
            "source_domain": meta.get("source_domain", ""),
        })

    return results