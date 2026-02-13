from scripts.utils import load_and_preprocess_docs, normalize_results, reciprocal_rank_fusion
import chainlit as cl
from sentence_transformers import SentenceTransformer
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

@cl.on_chat_start
async def start():

    """ Load the preprocessed documents and initialize the models and vector store
      when the user starts a chat session. """
    
    msg = cl.Message(content="Initializing models and loading data...")
    await msg.send()
    
    # Load already preprocessed documents and store them in the user session
    docs = load_and_preprocess_docs()

    # Tokenize corpus for BM25
    tokenized_corpus = [doc['text'].split(" ") for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)

    # Store docs mapped by ID for easy retrieval by BM25
    doc_map = {i: doc for i, doc in enumerate(docs)}

    # Store BM25 retriever and doc map in the user session for later use
    cl.user_session.set("bm25", bm25)
    cl.user_session.set("doc_map", doc_map)

    # Initialize embedding and reranker models from Hugging Face and store them in the user session
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    reranker_model = SentenceTransformer(RERANKER_MODEL)
    cl.user_session.set("embedding_model", embedding_model)
    cl.user_session.set("reranker_model", reranker_model)

    # Initialize Groq client and store it in the user session
    groq_client = Groq()
    cl.user_session.set("groq_client", groq_client)

    # Initialize Pinecone index and store it in the user session
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    cl.user_session.set("vector_index", index)
    
    # Send a welcome message
    await cl.Message(content="Hello! Ask me financial info about Apple or Nvidia.").send()

@cl.on_message
async def main(message: cl.Message):

    query = message.content
    
    # Retrieve tools from session
    bm25 = cl.user_session.get("bm25")
    doc_map = cl.user_session.get("doc_map")
    vector_index = cl.user_session.get("vector_index")
    embedding_model = cl.user_session.get("embedding_model")
    reranker_model = cl.user_session.get("reranker_model")
    groq_client = cl.user_session.get("groq_client")

    # --- Step 1: Retrieval (Hybrid) ---
    with cl.Step(name="Retrieval", type="tool") as step:
        step.input = query
        
        # Vector Search (Pinecone)
        with torch.no_grad(): # Disable gradient calculations to save memory during embedding
            query_embedding = embedding_model.encode(query).tolist()
        vector_results = vector_index.query(
            vector=query_embedding, 
            top_k=20, 
            include_metadata=True
        )
        vector_docs = normalize_results(vector_results.matches)

        # Keyword Search (BM25)
        tokenized_query = query.split(" ")
        bm25_top_n = bm25.get_top_n(tokenized_query, list(doc_map.values()), n=20)
        bm25_docs = normalize_results(bm25_top_n)

        # Ensemble (Reciprocal Rank Fusion)
        retrieved_docs = reciprocal_rank_fusion(vector_docs, bm25_docs)
        
        step.output = f"Found {len(retrieved_docs)} documents before reranking."

    # --- Step 2: Reranking ---
    with cl.Step(name="Reranking", type="tool") as step:

        # Create query-document pairs for reranking
        pairs = [[query, doc['text']] for doc in retrieved_docs]
        
        # Get scores
        with torch.no_grad(): # Disable gradient calculations to save memory during reranking
            scores = reranker_model.predict(pairs)
            # If it returns a tensor, move to CPU immediately to save GPU memory
            if isinstance(scores, torch.Tensor):
                scores = scores.detach().cpu().numpy()
        for doc, score in zip(retrieved_docs, scores):
            doc['rerank_score'] = score
            
        # Sort by rerank score and take top 5 documents to use as context
        ranked_docs = sorted(retrieved_docs, key=lambda x: x['rerank_score'], reverse=True)
        top_docs = ranked_docs[:5]
        
        # Show text in UI for debugging
        step.output = "\n\n".join([f"[{d['metadata'].get('company')}] {d['text'][:150]}..." for d in top_docs])
    
    # --- Step 3: Answer Generation ---
    
    # Prepare Context
    context_text = "\n\n".join([
        f"Source ({doc['metadata'].get('company', 'Unknown')} {doc['metadata'].get('year', '')}):\n{doc['text']}" 
        for doc in top_docs
    ])
    
    system_prompt = (
        "You are a useful financial RAG assistant. Use the following context to answer the user's query. "
        "Always cite your sources by referring to the Company and Year. "
        "If the context does not contain the answer, say you don't know."
        f"\n\nContext:\n{context_text}"
    )

    # Call Groq API with Streaming
    msg = cl.Message(content="")
    await msg.send()

    stream = groq_client.chat.completions.create(
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
            await msg.stream_token(chunk.choices[0].delta.content)

    # --- Step 4: Attach Sources ---
    source_elements = []
    for i, doc in enumerate(top_docs):
        source_name = f"Ref {i+1} ({doc['metadata'].get('company')})"
        source_elements.append(
            cl.Text(name=source_name, content=doc['text'], display="side")
        )
    
    msg.elements = source_elements
    await msg.update()

    # Cleanup to free memory
    del pairs, scores, retrieved_docs
    gc.collect()
    torch.cuda.empty_cache()