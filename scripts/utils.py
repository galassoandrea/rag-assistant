from langchain_core.documents import Document
from datasets import load_dataset
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

def load_and_preprocess_docs():

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

    return docs