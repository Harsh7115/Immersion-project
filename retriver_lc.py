# langchain_retriever.py
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from typing import List
import numpy as np
from retriever import retrieve

class AWCIMRetriever(BaseRetriever):
    """Wraps your existing retrieve() into a LangChain retriever."""
    top_k: int = 5

    def _get_relevant_documents(self, query: str) -> List[Document]:
        results = retrieve(query, top_k=self.top_k)
        return [
            Document(
                page_content=r["full_text"],
                metadata={**r["metadata"], "score": r["score"]}
            )
            for r in results
        ]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)