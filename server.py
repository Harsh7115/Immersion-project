from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rag_pipeline import rag_answer
from rag_pipeline_lc import rag_answer_lc

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ask")
def ask(query: str):
    # Generate answer/context documents
    answer, results = rag_answer_lc(query)
    return {
        "query": query,
        "answer": answer,
        "resources": results  # <-- include snippets + metadata
    }

# @app.get("/ask")
# def ask(query: str):
#     # Generate answer/context documents
#     answer, results = rag_answer(query, top_k=5)
#     return {
#         "query": query,
#         "answer": answer,
#         "resources": results  # <-- include snippets + metadata
#     }