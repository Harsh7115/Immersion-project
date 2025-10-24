# langchain_retriever.py
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from typing import ClassVar, Dict, List
import numpy as np
from retriever import retrieve_internal, retrieve_external

class AWCIMRetriever(BaseRetriever):
    """Wraps your existing retrieve() into a LangChain retriever."""
    top_k: int = 5
    extStore: ClassVar[Dict[str, List[Dict]]] = {}

    def _get_relevant_documents(self, query: str) -> List[Document]:
        intRes = retrieve_internal(query, top_k=self.top_k)
        extRes = retrieve_external(query, top_k=self.top_k)

        self.extStore[query] = [
            {"title": r["title"], "url": r["url"], "score": r["score"]}
            for r in extRes
        ]

        return [
            Document(
                page_content=r["full_text"],
                metadata={**r["metadata"], "score": r["score"]}
            )
            for r in intRes
        ]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

    def pop_external_links(self, query: str):
        return self.extStore.pop(query, [])