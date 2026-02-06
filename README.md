# 🤖 FinSearchAI: a RAG Assistant for Financial Insights

## 📖 Overview
FinSearch AI is a specialized Retrieval-Augmented Generation (RAG) system designed to answer financial queries using professional-grade open-source models. While standard RAG systems often struggle with the "noise" and structural complexity of financial data, this project implements a custom data-engineering pipeline to ensure high-fidelity retrieval and reasoning.

The system is built to provide accurate, context-aware answers to financial questions by retrieving relevant data from the FiQA (Financial Q&A) dataset - a high-signal corpus of investment and banking discussions.

## 📜 Steps
The RAG assistant works in a series of steps:

1. Load the dataset from Hugging Face.
   
3. Extract the first 500 documents and preprocess them to remove noisy patterns, deleterious for the embedding process.

4. Split the preprocessed documents into chunks with a bit of overlap, embed them through an embedding model and store them on a vector store.
   
5. Perform a search through a LLM agent, which receives in input a query, embeds it, retrieves the most relevant documents stored in Pinecone and uses them as context to generate an appropriate answer for the query.

## 🛠️ Technical Details

- To implement the RAG assistant, the main tools used are python and the langchain framework.

- The dataset used is the BeIR/FiQA dataset from Hugging Face.
  
- The model used to generate the documents' embeddings is BAAI/bge-m3, via Hugging Face Inference Endpoints.

- The brain (LLM) used for answering the queries is llama-3.3-70b-versatile, via the Groq API.
  
- To store the documents embeddings and perform high-speed semantic search, Pinecone (Cloud Vector DB) has been used.

## 🔎 Usage instructions

To test the system you can use the following queries:

- what is the purpose of a job training ?

- fdic savings account

- what is paypal currency value ?

- is netflix a dvd rental ?

- canadian importers trade in cny

You can find other queries to test on https://huggingface.co/datasets/BeIR/fiqa-generated-queries by looking at the query field. Keep in mind that, since only the first 500 document embeddings are stored in the vector store, some of the queries could not be answered.