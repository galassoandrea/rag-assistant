from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from utils import load_and_preprocess_docs

# Load already preprocessed documents
docs = load_and_preprocess_docs()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
docs = text_splitter.split_documents(docs)

# Initialize embedding model from Hugging Face
embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="BAAI/bge-m3",
)

# Store embeddings in Pinecone vector store
vectorstore = PineconeVectorStore.from_documents(
    documents=docs,
    index_name="rag-assistant-project",
    embedding=embeddings,
)
print("Database created and saved to Pinecone.")