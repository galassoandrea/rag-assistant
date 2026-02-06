from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from dotenv import load_dotenv

def retrieve_relevant_docs(query: str):

    # Initialize embedding model from Hugging Face
    embeddings = HuggingFaceEndpointEmbeddings(
        repo_id="BAAI/bge-m3",
    )
    
    # Initialize Pinecone vector store
    vectorstore = PineconeVectorStore(
        index_name="rag-assistant-project",
        embedding=embeddings
    )

    # Retrieve relevant documents using similarity search
    retrieved_docs = vectorstore.similarity_search(query, k=5)
    return retrieved_docs

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:

    # Extract query
    last_query = request.state["messages"][-1].text

    # Retrieve relevant documents, join them and inject them into a prompt
    retrieved_docs = retrieve_relevant_docs(last_query)
    print(retrieved_docs)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    system_message = (
        "You are a helpful assistant. Use the following context to provide an appropriate answer for the next query:"
        f"\n\n{docs_content}"
    )

    return system_message


def ask_question(query):
    model = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )
    agent = create_agent(model, tools=[], middleware=[prompt_with_context])
    for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        last_message = step["messages"][-1].text
        print(last_message)

if __name__ == "__main__":
    load_dotenv()
    ask_question("canadian importers trade in cny")
