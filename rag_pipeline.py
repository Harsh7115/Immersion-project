# rag_pipeline.py
import os
import time
from huggingface_hub import InferenceClient

# 1. Get HF token from environment
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set. Please export your Hugging Face token.")

# 2. Choose a small instruct-tuned model (good for free tier)
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# 3. Create the inference client
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

def rag_answer(query, retrieved_docs, max_retries=3):
    """
    Takes user query + retrieved docs, builds a prompt, and queries LLM.
    retrieved_docs: list of dicts with at least "full_text"
    """

    # Concatenate top-k docs into context
    context = "\n\n".join([doc.get("full_text", "") for doc in retrieved_docs])

    # Build prompt
    prompt = f"""You are a helpful assistant.
I will provide some context from documents and then a question.
Answer concisely and only based on the context.

Context:
{context}

Question: {query}
Answer:
"""

    # Call Hugging Face inference with retries
    for attempt in range(1, max_retries + 1):
        try:
            response = client.text_generation(prompt, max_new_tokens=300)
            return response.strip()

        except Exception as e:
            # Log the error (prints in your terminal)
            print(f"[RAG Pipeline] Error on attempt {attempt}: {repr(e)}")
            if "503" in str(e) and attempt < max_retries:
                print("Retrying after 3 seconds...")
                time.sleep(3)
                continue
            return f"Error from LLM: {repr(e)}"

    return "Error: max retries exceeded."
