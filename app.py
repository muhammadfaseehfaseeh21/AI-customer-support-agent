import os
import streamlit as st
from crewai import Agent, LLM
from crewai.tools import tool

# Page Configuration
st.set_page_config(
    page_title="AI Customer Support Hub",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ AI Customer Support Hub")
st.caption("Powered by Single Agent CrewAI & FAISS RAG")

# Helper function for Knowledge Tool
def create_knowledge_tool(vector_store):
    @tool("Search Order Knowledge Base")
    def search_tool(query: str) -> str:
        """Search the FAISS vector store for customer orders, addresses, and status."""
        if not vector_store:
            return "No vector store found."
        docs = vector_store.similarity_search(query, k=3)
        return "\n\n".join([d.page_content for d in docs])
    return search_tool

# API Key Handling
api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    api_key = st.sidebar.text_input("Enter Groq API Key:", type="password")

if not api_key:
    st.info("Please enter your Groq API Key in the sidebar to continue.")
    st.stop()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if user_prompt := st.chat_input("Ask about Customer ID, Order Date, Status, or Address..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Process Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("🤖 Agent analyzing query and searching records..."):
            try:
                # 1. Setup Retrieval Tool
                vector_store = st.session_state.get("vector_store", None)
                search_tool = create_knowledge_tool(vector_store)

                # 2. Configure LLM for Groq
                llm = LLM(
                    model="llama-3.3-70b-versatile",
                    provider="groq",
                    api_key=api_key
                )

                # 3. Create CrewAI Support Agent
                support_agent = Agent(
                    role="Customer Support Assistant",
                    goal="Provide clear, friendly, accurate information regarding order details, delivery dates, contact info, and addresses.",
                    backstory="You are a helpful, beginner-friendly e-commerce support specialist. You always retrieve factual order details from the database tool before answering.",
                    tools=[search_tool],
                    llm=llm,
                    verbose=True
                )

                # Execute task
                response = support_agent.execute_task(user_prompt)
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": str(response)})

            except Exception as e:
                st.error(f"An error occurred while processing: {str(e)}")
