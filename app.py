from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_community.retrievers import BM25Retriever
from scripts.EnsembleRetriever import EnsembleRetriever
from scripts.utils import load_and_preprocess_docs
import chainlit as cl

def retrieve_relevant_docs(query: str):
    
    # Wrap the retrieval of relevant documents in a Chainlit Step, so it appears in the UI as 'Retrieving Context'.
    with cl.Step(name="Retrieving Context", type="tool") as step:
        step.input = query

        # Create an ensemble retriever that combines BM25 and Vector search
        retriever = EnsembleRetriever(
            vector_retriever=cl.user_session.get("vectorstore").as_retriever(),
            bm25_retriever=cl.user_session.get("bm25_retriever"),
            top_k=20
        )
        retrieved_docs = retriever.invoke(query)

        # Rerank retrieved documents
        reranker = cl.user_session.get("reranker")
        
        # Create pairs of query and document for reranking
        pairs = [[query, doc.page_content] for doc in retrieved_docs]
        
        # Get reranking scores for each pair query-document
        scores = reranker.score(pairs)

        # Sort documents based on their scores
        scored_docs = sorted(
            zip(retrieved_docs, scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Select the top 5 documents after reranking
        final_retrieved_docs = [doc for doc, score in scored_docs[:5]]

        # Save the docs to the session so we can show them in the UI later
        cl.user_session.set("retrieved_docs", final_retrieved_docs)

        # Show the retrieved text in the UI step output
        step.output = "\n\n".join([d.page_content[:200] + "..." for d in retrieved_docs])

        return retrieved_docs

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:

    # Extract query
    last_query = request.state["messages"][-1].text

    # Retrieve relevant documents, join them and inject them into a prompt
    retrieved_docs = retrieve_relevant_docs(last_query)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    system_message = (
        "You are a helpful assistant. Use the following context to provide an appropriate answer for the next query:"
        f"\n\n{docs_content}"
    )

    return system_message

@cl.on_chat_start
async def start():

    """ Load the preprocessed documents and initialize the models and vector store
      when the user starts a chat session. """
    
    # Load already preprocessed documents and store them in the user session
    docs = load_and_preprocess_docs()
    cl.user_session.set("docs", docs)

    # Initialize BM25 retriever and store it in the user session
    bm25_retriever=BM25Retriever.from_documents(docs)
    cl.user_session.set("bm25_retriever", bm25_retriever)

    # Initialize embedding model from Hugging Face
    embeddings = HuggingFaceEndpointEmbeddings(
        repo_id="BAAI/bge-m3",
    )
            
    # Initialize Pinecone vector store and store it in the user session
    vectorstore = PineconeVectorStore(
        index_name="rag-assistant-project",
        embedding=embeddings
    )
    cl.user_session.set("vectorstore", vectorstore)

    # Initialize the Reranker (this downloads the model from HF and runs it locally on your CPU)
    reranker = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    cl.user_session.set("reranker", reranker)

    # Initialize the chat model
    model = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )
    
    # Create the agent and store it in the user session, so we don't reload it every message
    agent = create_agent(model, tools=[], middleware=[prompt_with_context])
    cl.user_session.set("agent", agent)
    
    # Send a welcome message
    await cl.Message(content="Hello! Ask me anything about financial imports.").send()

@cl.on_message
async def main(message: cl.Message):

    agent = cl.user_session.get("agent")
    
    # Create an empty message to stream the answer into
    msg = cl.Message(content="")
    await msg.send()

    # Iterate through the stream of events
    async for step in agent.astream(
        {"messages": [{"role": "user", "content": message.content}]},
        stream_mode="values",
    ):
        # Get the latest message content
        last_message = step["messages"][-1]
        
        # Only update if it's an AI message (not the user's input echoed back)
        if last_message.type == "ai" or (hasattr(last_message, "role") and last_message.role == "assistant"):
             msg.content = last_message.content
             await msg.update()
        
    # Get retrieved documents
    retrieved_docs = cl.user_session.get("retrieved_docs")
    print(retrieved_docs)
    
    if retrieved_docs:
        # Convert documents into Chainlit Text elements
        elements = [
            cl.Text(
                name=f"Source {i+1}", 
                content=doc.page_content, 
                display="side"
            )
            for i, doc in enumerate(retrieved_docs)
        ]
        msg.elements = elements
    
    # Final update to ensure formatting is perfect
    await msg.update()
