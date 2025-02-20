import os
import argparse
from typing import TypedDict
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from langchain_ollama import OllamaLLM

# Suppress Hugging Face tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration
EMBEDDING_MODEL = "BAAI/bge-base-en"  
LLM_NAME = "llama3.2:latest"

class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    llm = OllamaLLM(model=LLM_NAME)
    return {"messages": [llm.invoke(state["messages"])]}


def build_graph():
    graph_builder = StateGraph(State)
    memory = MemorySaver()
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph = graph_builder.compile(checkpointer=memory)
    return graph


def query_rag(question: str, graph: StateGraph, vector_store):

    # Find relevant docs in vector store
    relevant_docs = vector_store.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    prompt = (
        "You are an AI assistant. Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )
    
    config = {"configurable": {"thread_id": "1"}}
    events = graph.stream(
        {"messages": [{"role": "user", "content": prompt}]},
        config,
        stream_mode="updates",
    )

    for event in events:
        response_text = "\n".join(event['chatbot']['messages'])
        print(f"\n================ Chatbot ================\n\n{response_text}")

    return None


# CLI loop
def chat(database_dir: str, print_user_input: bool=False):

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.load_local(database_dir, embeddings, allow_dangerous_deserialization=True)

    graph = build_graph()

    print("\nWhat would you like to know about your lab's notebooks? (Type 'exit' to quit.)")
    while True:
        user_separator = "\n================== You ==================\n\n"
        user_input = input(user_separator)
        if user_input.lower() == "exit":
            break
        if print_user_input: print(user_separator, user_input)
        query_rag(user_input, graph, vector_store)

# Argparse
parser = argparse.ArgumentParser(description="Query a lab notebook vector database.")
parser.add_argument("--database", "-d",
                    required=True,
                    help="Path to directory containing lab notebook vector database.")

if __name__ == "__main__":
    args = parser.parse_args()
    chat(args.database)
