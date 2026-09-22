import os
import json
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from crewai.tools import tool

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "knowledge_base.json")

def prepare_vector_store():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Knowledge file not found at {DATA_PATH}")

    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    documents = []
    for item in data:
        content = (
            f"Customer ID: {item.get('customer_id', 'N/A')}\n"
            f"Order Date: {item.get('order_date', 'N/A')}\n"
            f"Contact Number: {item.get('contact_number', 'N/A')}\n"
            f"Status: {item.get('status', 'N/A')}\n"
            f"Shopping Address: {item.get('shopping_address', 'N/A')}\n"
        )
        documents.append(Document(page_content=content))

    # Free open-source embeddings (No OpenAI API Key needed)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def create_knowledge_tool(vector_store):
    @tool("Search Customer Order Database")
    def search_orders(query: str) -> str:
        """Searches customer records for ID, status, order date, contact, and address."""
        results = vector_store.similarity_search(query, k=2)
        if not results:
            return "No matching customer records found."
        return "\n---\n".join([doc.page_content for doc in results])

    return search_orders
