from scripts.utils import load_and_preprocess_docs, normalize_results, reciprocal_rank_fusion
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from groq import Groq
from dotenv import load_dotenv
import os
from pinecone import Pinecone
import torch
import gc

load_dotenv()

# Configuration
INDEX_NAME = "rag-assistant-v2"
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

st.set_page_config(page_title="Financial RAG Assistant", layout="wide")
st.title("🤖 Financial RAG Assistant")

# Initialization (Cached for performance)
@st.cache_resource
def initialize_models():
    """Load heavy models once and cache them across sessions."""
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    reranker_model = CrossEncoder(RERANKER_MODEL)
    return embedding_model, reranker_model

@st.cache_resource
def load_data():
    """Load and index documents for BM25 once."""
    docs = load_and_preprocess_docs()
    tokenized_corpus = [doc['text'].split(" ") for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    doc_map = {i: doc for i, doc in enumerate(docs)}
    return docs, bm25, doc_map

# Load resources
embedding_model, reranker_model = initialize_models()
docs, bm25, doc_map = load_data()

# Initialize API Clients in session state
if "groq_client" not in st.session_state:
    st.session_state.groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
if "vector_index" not in st.session_state:
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    st.session_state.vector_index = pc.Index(INDEX_NAME)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_retrieved" not in st.session_state:
    st.session_state.last_retrieved = []

# Initialize sidebar to display retrieved documents
with st.sidebar:
    st.header("📄 Retrieved Sources")
    if st.session_state.last_retrieved:
        for i, doc in enumerate(st.session_state.last_retrieved):
            with st.expander(f"Ref {i+1}: {doc['metadata'].get('company')} ({doc['metadata'].get('year')})"):
                st.write(f"**Score:** {doc.get('rerank_score', 'N/A'):.4f}")
                st.write(doc['text'])
    else:
        st.info("Sources will appear here after you ask a question.")

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if query := st.chat_input("Ask about Apple or Nvidia financial info..."):
    # Add user message to state and UI
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        # Create placeholders for status and response
        status_placeholder = st.empty()
        response_placeholder = st.empty()
        
        # --- Step 1 & 2: Retrieval & Reranking ---
        with status_placeholder.status("Searching and Reranking...", expanded=False) as status:
            # Vector Search
            with torch.no_grad():
                query_embedding = embedding_model.encode(query).tolist()
            vector_results = st.session_state.vector_index.query(
                vector=query_embedding, top_k=20, include_metadata=True
            )
            vector_docs = normalize_results(vector_results.matches)

            # BM25 Search
            tokenized_query = query.split(" ")
            bm25_top_n = bm25.get_top_n(tokenized_query, list(doc_map.values()), n=20)
            bm25_docs = normalize_results(bm25_top_n)

            # Fusion & Reranking
            retrieved_docs = reciprocal_rank_fusion(vector_docs, bm25_docs)
            pairs = [[query, doc['text']] for doc in retrieved_docs]
            
            with torch.no_grad():
                scores = reranker_model.predict(pairs)
                if isinstance(scores, torch.Tensor):
                    scores = scores.detach().cpu().numpy()
            
            for doc, score in zip(retrieved_docs, scores):
                doc['rerank_score'] = score
            
            top_docs = sorted(retrieved_docs, key=lambda x: x['rerank_score'], reverse=True)[:5]
            
            # Update session state for sidebar display
            st.session_state.last_retrieved = top_docs
            status.update(label="Information retrieved!", state="complete")

        # --- Step 3: Answer Generation ---
        context_text = "\n\n".join([
            f"Source ({d['metadata'].get('company')} {d['metadata'].get('year')}):\n{d['text']}" 
            for d in top_docs
        ])
        
        system_prompt = (
            "You are a useful financial RAG assistant. Use the following context to answer the user's query. "
            "Always cite your sources. If you don't know, say so.\n\n"
            f"Context:\n{context_text}"
        )

        full_response = ""
        stream = st.session_state.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
        
        # Save history and trigger a sidebar refresh
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # Cleanup
        del pairs, scores, retrieved_docs
        gc.collect()
        torch.cuda.empty_cache()
        st.rerun() # Refresh to update the sidebar with new sources