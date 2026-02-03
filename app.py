from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from dotenv import load_dotenv
import os

def retrieve_relevant_docs(query: str):

    # Load the embedding model and the vector store
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    vectorstore = PineconeVectorStore(
        index_name="rag-assistant-project",
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        embedding=embeddings
    )

    # Retrieve relevant documents using similarity search
    retrieved_docs = vectorstore.similarity_search(query, k=3)
    return retrieved_docs

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:

    # Extract query
    last_query = request.state["messages"][-1].text

    # Retrieve relevant documents, join them and inject them into a prompt
    retrieved_docs = retrieve_relevant_docs(last_query)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    system_message = (
        "You are a helpful assistant. Use the following context to answer the next question:"
        f"\n\n{docs_content}"
    )

    return system_message


def ask_question(query):
    model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview",
                                   google_api_key=os.getenv("GOOGLE_API_KEY"),
                                   temperature=0)
    agent = create_agent(model, tools=[], middleware=[prompt_with_context])
    for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        last_message = step["messages"][-1].text
        print(last_message)

if __name__ == "__main__":
    load_dotenv()
    ask_question("What is Tesla's approach to autonomous driving?")
