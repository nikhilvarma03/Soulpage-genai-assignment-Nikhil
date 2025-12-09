"""
Conversational Knowledge Bot (CKB)
A Streamlit chatbot that uses LangChain AgentExecutor with ConversationBufferMemory
to provide real-time, factual information while maintaining conversation context.
"""

import os
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferMemory
from ddgs import DDGS

# Current date for search enhancement
CURRENT_YEAR = datetime.now().year
TODAY_DATE = datetime.now().strftime("%B %d, %Y")
# Load environment variables (override=True ensures .env takes precedence)
load_dotenv(override=True)

# Page configuration
st.set_page_config(
    page_title="Conversational Knowledge Bot",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for styling the chat input (blue/teal theme)
st.markdown("""
<style>
    /* Chat input container */
    .stChatInput > div {
        border-color: #0ea5e9 !important;
    }

    /* Chat input text box */
    .stChatInput input {
        border-color: #0ea5e9 !important;
    }

    .stChatInput input:focus {
        border-color: #06b6d4 !important;
        box-shadow: 0 0 0 2px rgba(6, 182, 212, 0.2) !important;
    }

    /* Send button */
    .stChatInput button {
        background-color: #0ea5e9 !important;
        border-color: #0ea5e9 !important;
    }

    .stChatInput button:hover {
        background-color: #0284c7 !important;
        border-color: #0284c7 !important;
    }

    /* User message bubble */
    .stChatMessage[data-testid="user-message"] {
        background-color: #e0f2fe !important;
    }

    /* Assistant message bubble */
    .stChatMessage[data-testid="assistant-message"] {
        background-color: #f0fdf4 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Conversational Knowledge Bot")
st.caption("Ask me anything! I can search the web for real-time information.")

SYSTEM_PROMPT = f"""You are a helpful AI assistant with access to web search for real-time information.

TODAY'S DATE: {TODAY_DATE}
CURRENT YEAR: {CURRENT_YEAR}

CRITICAL RULES:
1. You MUST use the Search tool for ANY question about:
   - People (CEOs, leaders, celebrities, politicians, athletes, etc.)
   - Companies, organizations, or institutions
   - Current events, news, sports results, or recent developments
   - Facts that may have changed since your training data
   - Anything where accuracy and recency matters

2. When searching, include "{CURRENT_YEAR}" in your query for recent events.

3. Only skip the search for:
   - Basic greetings ("hello", "how are you")
   - Simple math or logic questions
   - Questions about today's date (answer: {TODAY_DATE})
   - Timeless facts (e.g., "what is the speed of light")

4. ALWAYS trust the search results over your training data - your knowledge is outdated.

5. After searching, synthesize the information and provide a clear, accurate answer.

6. Use the chat history for context in follow-up questions."""


def get_openai_api_key():
    """Get OpenAI API key from various sources."""
    # Try streamlit secrets first
    try:
        if 'OPENAI_API_KEY' in st.secrets:
            return st.secrets['OPENAI_API_KEY']
    except Exception:
        pass  # No secrets file exists
    # Then try environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        return api_key
    return None


def create_search_tool():
    """Create an enhanced DuckDuckGo search tool using ddgs package."""

    def search_web(query: str) -> str:
        """
        Enhanced web search using DuckDuckGo.
        Searches both news and web results for comprehensive coverage.
        """
        try:
            output = []
            seen_content = set()  # Avoid duplicate content

            with DDGS() as ddgs:
                # 1. NEWS SEARCH - Best for recent events, sports, current affairs
                try:
                    news_results = list(ddgs.news(query, max_results=5))
                    if news_results:
                        output.append("=== RECENT NEWS ===")
                        for r in news_results:
                            title = r.get('title', '')
                            body = r.get('body', '')
                            date = r.get('date', '')[:10] if r.get('date') else ''
                            source = r.get('source', '')
                            content_key = body[:100] if body else title
                            if content_key not in seen_content:
                                seen_content.add(content_key)
                                output.append(f"[{date}] ({source}) {title}\n{body}")
                except Exception:
                    pass  # Continue even if news search fails

                # 2. WEB SEARCH - Good for general information
                try:
                    text_results = list(ddgs.text(query, max_results=5))
                    if text_results:
                        output.append("\n=== WEB RESULTS ===")
                        for r in text_results:
                            title = r.get('title', '')
                            body = r.get('body', '')
                            href = r.get('href', '')
                            content_key = body[:100] if body else title
                            if content_key not in seen_content:
                                seen_content.add(content_key)
                                output.append(f"{title}\n{body}\nSource: {href}")
                except Exception:
                    pass  # Continue even if text search fails

                # 3. If query seems like it needs current year, search again with year
                if not any(str(CURRENT_YEAR) in str(r) for r in output):
                    try:
                        year_query = f"{query} {CURRENT_YEAR}"
                        year_results = list(ddgs.news(year_query, max_results=3))
                        if year_results:
                            output.append(f"\n=== {CURRENT_YEAR} SPECIFIC RESULTS ===")
                            for r in year_results:
                                title = r.get('title', '')
                                body = r.get('body', '')
                                date = r.get('date', '')[:10] if r.get('date') else ''
                                content_key = body[:100] if body else title
                                if content_key not in seen_content:
                                    seen_content.add(content_key)
                                    output.append(f"[{date}] {title}\n{body}")
                    except Exception:
                        pass

            if not output:
                return f"No search results found for '{query}'. Try rephrasing or adding more context."

            return "\n\n".join(output)

        except Exception as e:
            return f"Search error: {str(e)}. Please try a different query."

    return Tool(
        name="Search",
        func=search_web,
        description=f"Search the internet for current events, news, people, companies, sports results, and factual information. For recent events, the tool automatically includes {CURRENT_YEAR}. Input should be a clear search query."
    )


def initialize_agent(api_key: str):
    """Initialize the LangChain AgentExecutor with ConversationBufferMemory."""
    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        api_key=api_key
    )

    # Create search tool
    search_tool = create_search_tool()
    tools = [search_tool]

    # Create the prompt template with memory placeholders
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create ConversationBufferMemory to remember previous conversations
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Create the agent using OpenAI functions
    agent = create_openai_functions_agent(llm, tools, prompt)

    # Create AgentExecutor with memory
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent_executor


def get_agent_response(agent_executor, user_input: str) -> str:
    """Get response from AgentExecutor with error handling.

    The AgentExecutor with ConversationBufferMemory automatically handles
    conversation history, so we just need to pass the user input.
    """
    try:
        # Invoke agent - memory is handled automatically by ConversationBufferMemory
        response = agent_executor.invoke({"input": user_input})
        return response.get("output", "I apologize, but I couldn't generate a response.")
    except Exception as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower():
            return "I'm currently experiencing high demand. Please try again in a moment."
        elif "api key" in error_msg.lower() or "authentication" in error_msg.lower():
            # Clear cached agent so it reinitializes with fresh env vars
            if "agent" in st.session_state:
                del st.session_state.agent
            return "API key error. Please refresh the page after updating your API key."
        else:
            return "I encountered an error while processing your request. Please try again."


def main():
    """Main application logic."""
    # Check for API key
    api_key = get_openai_api_key()

    if not api_key:
        st.warning("⚠️ OpenAI API key not found!")
        st.info("""
        Please provide your OpenAI API key using one of these methods:
        1. Create a `.env` file with `OPENAI_API_KEY=your_key_here`
        2. Set the `OPENAI_API_KEY` environment variable
        3. Add it to Streamlit secrets (`.streamlit/secrets.toml`)
        """)

        # Allow manual input as fallback
        manual_key = st.text_input("Or enter your API key here:", type="password")
        if manual_key:
            api_key = manual_key
        else:
            st.stop()

    # Initialize session state for messages display
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize AgentExecutor with ConversationBufferMemory in session state
    # Memory is built into the agent, so we don't need separate message tracking
    if "agent" not in st.session_state:
        with st.spinner("Initializing the knowledge bot..."):
            st.session_state.agent = initialize_agent(api_key)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if user_input := st.chat_input("Ask me anything..."):
        # Add user message to display history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_agent_response(
                    st.session_state.agent,
                    user_input
                )
                st.markdown(response)

        # Add assistant response to display history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.markdown("""
        **Conversational Knowledge Bot** uses:
        - 🧠 LangChain AgentExecutor
        - 💾 ConversationBufferMemory
        - 🔍 DuckDuckGo for web search

        **Features:**
        - Real-time web search
        - Conversation memory
        - Follow-up questions
        """)

        st.divider()

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            # Clear the ConversationBufferMemory
            if "agent" in st.session_state:
                st.session_state.agent.memory.clear()
            st.rerun()

        st.divider()
        st.caption("Powered by OpenAI & LangChain")


if __name__ == "__main__":
    main()
