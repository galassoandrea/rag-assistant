from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List

class EnsembleRetriever(BaseRetriever):
    """
    A custom retriever that combines BM25 (keyword) and Vector (semantic) search
    using Reciprocal Rank Fusion (RRF).
    """
    # Define the two retrievers
    vector_retriever: BaseRetriever
    bm25_retriever: BaseRetriever
    c: int = 60  # RRF constant
    top_k: int = 5  # Number of top documents to return
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        
        # Get results from both retrievers
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        # RRF Algorithm: Fuse the rankings
        # Map each document's content/ID to a score
        rrf_score = {}
        
        # Helper to process a list of docs
        def process_docs(docs):
            for rank, doc in enumerate(docs):
                # Create a unique key (content or ID) to deduplicate
                doc_key = doc.page_content 
                if doc_key not in rrf_score:
                    rrf_score[doc_key] = {"doc": doc, "score": 0.0}
                
                # RRF Formula: 1 / (k + rank)
                rrf_score[doc_key]["score"] += 1.0 / (self.c + rank + 1)

        process_docs(vector_docs)
        process_docs(bm25_docs)

        # Sort documents by combined score (highest first)
        sorted_docs = sorted(
            rrf_score.values(), key=lambda x: x["score"], reverse=True
        )
        
        # Return the top documents
        return [item["doc"] for item in sorted_docs[:self.top_k]]