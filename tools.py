import json
import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from crewai.tools import tool

DATA_PATH = "data/orders_knowledge.json"

def prepare_vector_store(openai_api_key: str):
    """Loads JSON data, creates text chunks, builds embeddings, and creates FAISS index."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Knowledge file not found at {DATA_PATH}")

    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    # Convert structured order entries into readable text documents
    documents = []
    for item in data:
        content = (
            f"Customer ID: {item['customer_id']}\n"
            f"Order Date: {item['order_date']}\n"
            f"Contact Number: {item['contact_number']}\n"
            f"Status: {item['status']}\n"
            f"Shopping Address: {item['shopping_address']}\n"
            f"Items Purchased: {item['items']}\n"
            f"Total Amount: {item['total_amount']}\n"
        )
        documents.append(Document(page_content=content, metadata={"customer_id": item['customer_id']}))

    # Chunking long texts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Generating Embeddings & FAISS Vector DB
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

def create_knowledge_tool(vector_store):
    """Creates a custom CrewAI tool for vector similarity retrieval."""
    
    @tool("Search Customer Order Database")
    def search_orders(query: str) -> str:
        """Searches the knowledge base for customer details, order status, order date, contact number, and shipping address."""
        results = vector_store.similarity_search(query, k=2)
        if not results:
            return "No matching customer or order records found."
        return "\n---\n".join([doc.page_content for doc in results])

    return search_orders
