import os
import streamlit as st
from crewai import Agent, Task, Crew, LLM
from rag_utils import prepare_vector_store, create_knowledge_tool

# Streamlit Page Config
st.set_page_config(
    page_title="SupportAI - Customer Hub",
    page_icon="🛍️",
    layout="centered"
)

# Vivid Gradient Theme styling
CUSTOM_CSS = """
<style>
    .stApp {
        background: linear-gradient(135deg, #1A1C29 0%, #0F2027 50%, #203A43 100%);
        color: #FFFFFF;
    }
    .main-card {
        background: rgba(255, 255, 255, 0.08);
        padding: 25px;
        border-radius: 16px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        margin-bottom: 20px;
    }
    .badge {
        background: linear-gradient(90deg, #FF416C, #FF4B2B);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        color: white;
    }
    .stButton>button {
        background: linear-gradient(90deg, #11998e, #38ef7d) !important;
        color: black !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        border: none !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Secure API Key handling via Streamlit Secrets or Sidebar Input
api_key = st.secrets.get("GROQ_API_KEY") if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")

with st.sidebar:
    st.header("⚙️ Configuration")
    if not api_key:
        api_key = st.text_input("Enter API Key:", type="password")
    
    st.markdown("---")
    st.markdown("### 📋 Quick Demo Queries")
    st.code("What is the status of order for CUST-1002?")
    st.code("Show shopping address & contact for CUST-1001")
    st.code("When was CUST-1003's order placed?")

# Main Header UI
st.markdown("""
<div class="main-card">
    <h1>🛍️ AI Customer Support Hub</h1>
    <p>Powered by <span class="badge">Single Agent CrewAI</span> & <span class="badge">FAISS RAG</span></p>
</div>
""", unsafe_allow_html=True)

if not api_key:
    st.warning("⚠️ Please provide an API key in Streamlit secrets or via the sidebar to start.")
    st.stop()

# Initialize Vector Store into Session State
if "vector_store" not in st.session_state:
    with st.spinner("⏳ Creating Chunks & FAISS Embeddings..."):
        try:
            st.session_state.vector_store = prepare_vector_store()
            st.success("✅ Knowledge base indexed successfully!")
        except Exception as e:
            st.error(f"Error loading knowledge base: {e}")
            st.stop()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process User Input
if user_query := st.chat_input("Ask about Customer-ID, Order Date, Status, or Address..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Agent analyzing query and searching records..."):
            try:
                # 1. Setup Retrieval Tool
                search_tool = create_knowledge_tool(st.session_state.vector_store)

                # 2. Configure LLM with openai/gpt-oss-120b
                llm = LLM(
                    model="groq/openai/gpt-oss-120b",
                    api_key=api_key,
                    base_url="https://api.groq.com/openai/v1"
                    cache_prompt=False
                )

                # 3. Create Single CrewAI Agent
                support_agent = Agent(
                    role="Customer Support Assistant",
                    goal="Provide clear, friendly, accurate information regarding order details, delivery dates, contact info, and addresses.",
                    backstory="You are a helpful, beginner-friendly e-commerce support specialist. You always retrieve factual order details from the database tool before answering.",
                    tools=[search_tool],
                    llm=llm,
                    verbose=False
                )

                # 4. Create Single Task
                support_task = Task(
                    description=f"Answer the customer question accurately using your search tool: '{user_query}'",
                    expected_output="A helpful, precise, customer-friendly answer containing accurate order information (Customer ID, Status, Shipping Address, etc.) if requested.",
                    agent=support_agent
                )

                # 5. Execute Single Agent Crew
                crew = Crew(
                    agents=[support_agent],
                    tasks=[support_task]
                )
                
                result = crew.kickoff()
                response_text = str(result)

                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

            except Exception as e:
                st.error(f"An error occurred while processing: {e}")
