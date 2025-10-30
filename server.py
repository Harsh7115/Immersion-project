from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
# from rag_pipeline import rag_answer
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
def ask(query: str, request: Request):
    # Generate answer/context documents
    forwarded_for = request.headers.get("x-forwarded-for")
    client_ip = forwarded_for.split(",")[0] if forwarded_for else request.client.host
    print(client_ip, query)
    answer, uris, extUris, results = rag_answer_lc(query)
    return {
        "query": query,
        "answer": answer,
        "intLinks": uris,
        "extLinks": extUris,
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