from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from dotenv import load_dotenv
import chainlit as cl

# Initialize embedding model from Hugging Face
embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="BAAI/bge-m3",
)
        
# Initialize Pinecone vector store
vectorstore = PineconeVectorStore(
    index_name="rag-assistant-project",
    embedding=embeddings
)

def retrieve_relevant_docs(query: str):
    # Wrap the retrieval of relevant documents in a Chainlit Step, so it appears in the UI as 'Retrieving Context'.
    with cl.Step(name="Retrieving Context", type="tool") as step:
        step.input = query

        # Retrieve relevant documents using similarity search
        retrieved_docs = vectorstore.similarity_search(query, k=5)

        # Show the retrieved text in the UI step output
        step.output = "\n\n".join([d.page_content[:200] + "..." for d in retrieved_docs])

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

@cl.on_chat_start
async def start():
    # Initialize the agent when the user opens the page.
    model = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )
    
    # Create the agent and store it in the user session
    # so we don't reload it every message
    agent = create_agent(model, tools=[], middleware=[prompt_with_context])
    cl.user_session.set("agent", agent)
    
    # Send a welcome message
    await cl.Message(content="Hello! Ask me anything about financial imports.").send()

@cl.on_message
async def main(message: cl.Message):
    # Runs on every user message.
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
    
    # Final update to ensure formatting is perfect
    await msg.update()

#if __name__ == "__main__":
#    load_dotenv()
#    ask_question("canadian importers trade in cny")
