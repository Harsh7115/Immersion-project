# rag_langchain.py
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEndpoint
from langchain_aws import ChatBedrock
from retriver_lc import AWCIMRetriever
import os

# Instantiate your retriever
retriever = AWCIMRetriever(top_k=5)

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Hugging Face token not found. Please set HF_TOKEN in your environment variables.")

# Hugging Face LLM (can swap for local/Claude/OpenAI)
llm = ChatBedrock(
    model_id="arn:aws:bedrock:us-east-1:088640340910:inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="us-east-1",
    provider="anthropic",
    model_kwargs={
        "temperature": 0.7,
        "max_tokens": 300
    }
)

# Prompt template
template = """
You are a helpful assistant trained to support patients and caregivers.

Use the context below to answer the user's question accurately and compassionately.
If the context seems irrelevant, respond with “I'm not sure based on the available data.”

Context:
{context}

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
