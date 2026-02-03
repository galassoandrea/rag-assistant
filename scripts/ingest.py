from langchain_community.document_loaders import DirectoryLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

def ingest_docs(folder_path):
    # Load all HTML docs in the directory and eventual subdirectories
    loader = DirectoryLoader(
        "./data", 
        glob="**/*.html", 
        loader_cls=UnstructuredHTMLLoader
    )
    raw_docs = loader.load()
    print(f"Loaded {len(raw_docs)} pages from folder.")

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(raw_docs)

    # Initialize Gemini Embeddings
    load_dotenv()
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Store embeddings in Pinecone vector store
    vectorstore = PineconeVectorStore.from_documents(
        documents=docs,
        index_name="rag-assistant-project",
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    )
    print("Database created and saved to Pinecone.")

if __name__ == "__main__":
    ingest_docs("./data")