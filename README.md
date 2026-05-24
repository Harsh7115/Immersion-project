# Immersion Project — CanHeal AI Chatbot

A **RAG-based AI chatbot toolkit** built for the CanHeal Cancer oncology platform, helping patients and caregivers navigate cancer resources through natural language.

## Overview

CanHeal needed an intelligent assistant that could answer questions about oncology resources, treatment options, and support networks — without hallucinating or going off-topic. This project delivers a retrieval-augmented generation pipeline that grounds every response in trusted source documents.

## Features

- **RAG Pipeline** — embeds and indexes CanHeal's resource corpus; retrieves relevant chunks before generation
- **Conversational Memory** — multi-turn context management so follow-up questions resolve correctly
- **Source Attribution** — every answer cites the specific documents it drew from
- **Guardrails** — topic filtering to keep responses within the oncology/support domain
- **Streaming Responses** — token-by-token output for a responsive UX

## Architecture

```
User Query
    ↓
Query Embedding (text-embedding-ada-002)
    ↓
Vector Search (FAISS / Chroma)
    ↓
Top-K Chunk Retrieval
    ↓
Prompt Assembly [system + context + history + query]
    ↓
LLM Generation (GPT-4)
    ↓
Streamed Response + Source Citations
```

## Tech Stack

Python · LangChain · OpenAI API · FAISS · FastAPI · React

## Setup

```bash
git clone https://github.com/Harsh7115/Immersion-project
cd Immersion-project
pip install -r requirements.txt
cp .env.example .env  # add OPENAI_API_KEY
python ingest.py       # embed and index documents
uvicorn main:app --reload
```

## Key Design Decisions

- **Chunk size 512 tokens** with 64-token overlap — balances retrieval precision vs. context richness
- **Top-3 retrieval** — enough context without overwhelming the prompt
- **Temperature 0.3** — factual, grounded answers over creative generation
- System prompt enforces domain scope and mandates source citation

---

Built for the CanHeal Cancer platform to make oncology information more accessible through conversational AI.
