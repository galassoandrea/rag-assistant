import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from utils import load_and_preprocess_docs

load_dotenv()

# Configuration
INDEX_NAME = "rag-assistant-v2"
EMBEDDING_MODEL = "BAAI/bge-m3"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
BATCH_SIZE = 32

# Initialize Clients
pc = Pinecone(api_key=PINECONE_API_KEY)
model = SentenceTransformer(EMBEDDING_MODEL)

def generate_embeddings(texts):
    """
    Generates embeddings using Hugging Face Inference API.
    """
    try:
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=BATCH_SIZE)
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []

def main():
    # Get Pinecone index (already created through Pinecone UI)
    index = pc.Index(INDEX_NAME)

    # Load document chunks with metadata
    chunks = load_and_preprocess_docs()
    
    print("Starting embedding and upsert process...")

    texts = [item['text'] for item in chunks]
    embeddings = generate_embeddings(texts)

    # Prepare vectors for Pinecone upsert
    vectors_to_upsert = []
    for j, embedding in enumerate(embeddings):
        chunk_data = chunks[j]
        
        # Combine original metadata with the text content 
        meta = chunk_data['metadata'].copy()
        meta['text'] = chunk_data['text']
        
        vectors_to_upsert.append({
            "id": chunk_data['id'],
            "values": embedding,
            "metadata": meta
        })
        
    # Upsert to Pinecone
    try:
        index.upsert(vectors=vectors_to_upsert)
        print(f"Upserted batch {j} to {j + len(chunks)}")
    except Exception as e:
        print(f"Error upserting batch: {e}")

    print("Database created and saved to Pinecone.")

if __name__ == "__main__":
    main()