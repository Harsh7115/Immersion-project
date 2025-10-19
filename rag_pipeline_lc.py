# rag_langchain.py
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEndpoint
from langchain_aws import ChatBedrock
from retriver_lc import AWCIMRetriever
import os

# Instantiate your retriever
retriever = AWCIMRetriever(top_k=5)

# HF_TOKEN = os.environ.get("HF_TOKEN")
# if not HF_TOKEN:
#     raise ValueError("Hugging Face token not found. Please set HF_TOKEN in your environment variables.")

# Rajesh AWS Account
llm = ChatBedrock(
    model_id="arn:aws:bedrock:us-east-1:088640340910:inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="us-east-1",
    provider="anthropic",
    model_kwargs={
        "temperature": 0.7,
        "max_tokens": 300
    }
)

# AWCIM AWS Account
# llm = ChatBedrock(
#     model_id="anthropic.claude-3-sonnet-20240229-v1:0",  # ✅ On-demand model
#     region_name="us-east-1",
#     provider="anthropic",
#     model_kwargs={
#         "temperature": 0.7,
#         "max_tokens": 300,
#         "anthropic_version": "bedrock-2023-05-31"  # ✅ Required field for Anthropic 3.x
#     }
# )


# Prompt template
template = """
You are a caring and knowledgeable AI assistant trained to support patients and caregivers.

# === C (Context) ===
Use the background information below to guide your answer.
If the information does not include the needed details, give a gentle, general explanation instead of pointing out missing context.

Context:
{context}

# === o (Objective) ===
Provide an accurate, compassionate, and easy-to-understand answer to the user’s question.
If the question cannot be answered safely or clearly, respond politely with:
"Sorry, I cannot help with that question."

# === S (Style) ===
Warm, empathetic, conversational, and medically responsible.
Avoid phrases like “Based on the context provided” or “not detailed in this information.”
Do not refer to yourself or the source; speak directly to the user.

# === T (Audience) ===
Patients, caregivers, or anyone seeking reliable health and wellness guidance.

# === A (Response Format) ===
- Write 2–4 sentences.
- Speak naturally and reassuringly.
- Avoid repeating the question or describing the source.
- When information is limited, provide a brief general explanation in simple language.

Question:
{question}

Answer:

"""
prompt = PromptTemplate.from_template(template)

# Create the Retrieval-Augmented QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
)

def rag_answer_lc(query: str):
    """Return both answer and documents (sources)."""
    result = qa_chain.invoke({"query": query})
    print(result.keys())
    answer = result.get("result") or result.get("answer") or str(result)

    # Return the same tuple format as before
    return answer, result.get("source_documents", [])
