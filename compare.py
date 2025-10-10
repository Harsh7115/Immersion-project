# compare_embeddings.py
import numpy as np, pickle
from sentence_transformers import SentenceTransformer
from retriever import retrieve, chunks, chunk_to_doc_idx, documents, metadata, embeddings as mini_embs

# Load BGE model
bge_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Encode the same chunks with BGE (smaller subset for speed)
sample_chunks = chunks[:1000]
bge_embs = bge_model.encode(sample_chunks, convert_to_numpy=True, normalize_embeddings=True)

# Pick some test queries relevant to your domain
queries = [
    "What are the side effects of chemotherapy?",
    "How can I manage anxiety during cancer treatment?",
    "What are good dietary guidelines for caregivers?",
    "Are there any integrative medicine practices for fatigue?",
    "What to expect after radiation therapy?"
]

def evaluate_model(model_name, model_embs, query_encoder):
    recalls = []
    for q in queries:
        q_emb = query_encoder.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        sims = (model_embs @ q_emb.T).squeeze()
        top_k = np.argsort(sims)[::-1][:5]
        top_texts = [sample_chunks[i][:150] for i in top_k]
        print(f"\n🔹 Query ({model_name}): {q}")
        for i, t in enumerate(top_texts, 1):
            print(f"  {i}. {t.strip()}...")
        recalls.append(sims[top_k[0]])  # proxy metric
    print(f"\nAvg top-1 sim for {model_name}: {np.mean(recalls):.4f}\n")

# Run comparisons
print("=== MINI-LM BASELINE ===")
evaluate_model("MiniLM", mini_embs[:1000], SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"))

print("\n=== BGE MODEL ===")
evaluate_model("BGE", bge_embs, bge_model)
