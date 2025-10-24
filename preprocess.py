import json
import re
from pathlib import Path
import spacy
from sentence_transformers import SentenceTransformer, util

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

nlp = spacy.load("en_core_web_sm")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text(text, model=None, sim_threshold=0.7, max_clauses=4):
    mdl = model or embed_model
    """
    Hybrid semantic + propositional + structure-aware chunking.

    Steps:
      1. Split text by structural markers ([[Heading]] etc.)
      2. Tokenize sentences & clauses (spaCy)
      3. Embed clauses and merge semantically similar ones
      4. Add small overlap between consecutive chunks
    """
    # 1. Split by headings / structural markers
    sections = re.split(r"\[\[.*?\]\]", text)
    all_chunks = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # 2. Sentence and clause segmentation
        doc = nlp(section)
        clauses = []
        for sent in doc.sents:
            # Split sentences further by coordinating conjunctions
            for chunk in sent.text.split(","):
                clause = chunk.strip()
                if len(clause.split()) >= 4:
                    clauses.append(clause)

        if not clauses:
            continue

        # 3. Semantic grouping
        emb = mdl.encode(clauses, normalize_embeddings=True)
        cur_chunk, chunks = [clauses[0]], []
        for i in range(1, len(clauses)):
            sim = util.cos_sim(emb[i - 1], emb[i]).item()
            if sim < sim_threshold or len(cur_chunk) >= max_clauses:
                chunks.append(" ".join(cur_chunk))
                cur_chunk = []
            cur_chunk.append(clauses[i])
        if cur_chunk:
            chunks.append(" ".join(cur_chunk))

        # 4. Add 1-clause overlap between consecutive chunks
        overlapped = []
        for i, ch in enumerate(chunks):
            if i > 0:
                prev_last = chunks[i - 1].split()[-10:]
                ch = " ".join(prev_last) + " " + ch
            overlapped.append(ch)

        all_chunks.extend(overlapped)

    return all_chunks
