# rag_pipeline.py
import os
import time
from huggingface_hub import InferenceClient
from retriever import retrieve

import os

HF_TOKEN = os.environ.get("HF_TOKEN") or "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"
if not HF_TOKEN:
    raise ValueError("Hugging Face token not found. Please set HF_TOKEN in your environment variables.")

hf_client = InferenceClient(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    token=HF_TOKEN
)

def rag_answer(query, top_k=5):
    results = retrieve(query, top_k=top_k)
    context = "\n\n".join([r["snippet"] for r in results])

    # Build messages for conversational endpoint
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
    ]

    # Call Hugging Face Inference API for LLaMA 3 (chat completion)
    response = hf_client.chat_completion(
        messages=messages,
        max_tokens=300,
        temperature=0.7
    )

    # The answer will be inside choices[0].message
    return response.choices[0].message["content"], results
