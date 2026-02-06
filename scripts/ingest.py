from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document
from datasets import load_dataset
from dotenv import load_dotenv
import re
import html

def preprocess_docs(docs):
    # Decode HTML entities (e.g., &amp; -> &)
    docs = html.unescape(docs)
    
    # Remove URLs which are deleterious to embedding quality
    docs = re.sub(r'http\S+', '', docs)
    
    # Remove common FiQA/Reddit bot signatures
    noise_patterns = [
        r"PM's and comments are monitored",
        r"constructive feedback is welcome",
        r"Version \d+\.\d+, ~\d+ tl;drs so far",
        r"Top keywords:.*",
        r"Extended Summary | FAQ | Feedback",
        r"\[.*?\]\(.*?\)"
    ]
    for pattern in noise_patterns:
        docs = re.sub(pattern, "", docs, flags=re.IGNORECASE)
    
    # Clean up extra whitespace/newlines
    clean_docs = re.sub(r'\s+', ' ', docs).strip()
    
    return clean_docs

def ingest_docs(folder_path):

    # Load the FiQA dataset from huggingface
    dataset = load_dataset("BeIR/fiqa", "corpus", split="corpus")

    docs = []
    # Extract the first 500 rows
    for i in range(500):
        content = dataset[i]['text']
        # Preprocess documents to filter out noise before embedding
        content = preprocess_docs(content)
        metadata = {"id": dataset[i]['_id'], "title": dataset[i].get('title', "")}
        # Convert to Langchain Document format
        docs.append(Document(page_content=content, metadata=metadata))

    print(f"Loaded {len(docs)} documents.")

    for i in range(len(docs)):
        if "keywords" in docs[i].page_content:
            docs[i].page_content = ""

    # Split into chunks
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

if __name__ == "__main__":
    load_dotenv()
    ingest_docs("./data")