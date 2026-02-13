import os
import glob
import re
from bs4 import BeautifulSoup

def load_and_preprocess_docs():
    """
    Loads HTML documents, extracts text while preserving semantic structure and
    generates metadata for each chunk. The function implements an "aggregation" logic
    to create coherent chunks of text based on HTML tags, ensuring that each chunk is meaningful
    and not just a random split of characters.
    """
    print("Loading documents...")
    file_paths = glob.glob("data/**/*.html", recursive=True)
    processed_chunks = []

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')

            # Metadata Extraction
            metadata = {'source': file_path}
            ticker = re.search(r'sec-edgar-filings[\\/]([^\\/]+)', file_path)
            metadata['company'] = ticker.group(1).upper() if ticker else "UNKNOWN"
            
            acc_num = re.search(r'\d{10}-(\d{2})-\d{6}', file_path)
            if acc_num:
                metadata['year'] = 2000 + int(acc_num.group(1)) - 1 # Fiscal Year
            else:
                metadata['year'] = 0

            # The "Aggregation" Splitting Logic
            
            current_chunk_text = ""
            current_chunk_size = 0
            CHUNK_LIMIT = 2000
            
            # Define which HTML tags to consider for text extraction and chunking
            tags_to_parse = soup.find_all(['p', 'div', 'table', 'h1', 'h2', 'h3', 'h4', 'ul', 'ol'])
            
            for tag in tags_to_parse:
                # Get clean text from the tag
                tag_text = tag.get_text(separator=" ", strip=True)
                
                # Skip empty tags
                if not tag_text:
                    continue
                
                tag_len = len(tag_text)

                # Check if adding this tag would exceed the chunk limit
                if current_chunk_size + tag_len <= CHUNK_LIMIT:
                    current_chunk_text += "\n" + tag_text
                    current_chunk_size += tag_len
                else:
                    # Save the CURRENT bucket if it has content
                    if current_chunk_text:
                        processed_chunks.append({
                            "id": f"{os.path.basename(file_path)}_{len(processed_chunks)}",
                            "text": current_chunk_text.strip(),
                            "metadata": metadata
                        })
                    
                    # Start a NEW bucket with this tag
                    current_chunk_text = tag_text
                    current_chunk_size = tag_len

            # After processing all tags, save any remaining text as a final chunk
            if current_chunk_text:
                processed_chunks.append({
                    "id": f"{os.path.basename(file_path)}_{len(processed_chunks)}",
                    "text": current_chunk_text.strip(),
                    "metadata": metadata
                })

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"Generated {len(processed_chunks)} chunks.")
    return processed_chunks

def normalize_results(results):
    """
    Standardizes results from different retrievers (BM25 & Pinecone) 
    into a common format: [{'id': str, 'text': str, 'score': float, 'metadata': dict}]
    """
    normalized = []
    for doc in results:
        # Check if it's a Pinecone result (has 'values' or 'matches')
        if hasattr(doc, 'metadata'): # Pinecone object
            normalized.append({
                "id": doc.id,
                "text": doc.metadata.get('text', ''),
                "metadata": doc.metadata,
                "score": doc.score
            })
        elif isinstance(doc, dict): # BM25 result
            normalized.append(doc)
    return normalized

def reciprocal_rank_fusion(list_a, list_b, k=60):
    """
    Combines two lists of ranked documents using Reciprocal Rank Fusion (RRF).
    """
    fused_scores = {}
    doc_map = {} # Store the actual doc data

    # Process first list (Vector Results)
    for rank, doc in enumerate(list_a):
        doc_id = doc['id']
        if doc_id not in fused_scores:
            fused_scores[doc_id] = 0
            doc_map[doc_id] = doc
        fused_scores[doc_id] += 1 / (rank + k)

    # Process second list (BM25 Results)
    for rank, doc in enumerate(list_b):
        doc_id = doc['id']
        if doc_id not in fused_scores:
            fused_scores[doc_id] = 0
            doc_map[doc_id] = doc
        fused_scores[doc_id] += 1 / (rank + k)

    # Sort by fused score
    reranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return list of document objects for the top 20 results
    return [doc_map[doc_id] for doc_id, score in reranked_results[:20]]