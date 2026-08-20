from langchain_ollama import OllamaLLM
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.memory import ConversationBufferWindowMemory
from langchain import hub
from duckduckgo_search import DDGS
from rag_engine import retrieve_relevant_chunks


def search_web(query: str) -> str:
    # DuckDuckGo search - free, no API key needed
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No web results found for this query."
        formatted = []
        for r in results:
            formatted.append(
                f"Title: {r['title']}\n"
                f"URL: {r['href']}\n"
                f"Content: {r['body']}"
            )
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Web search encountered an error: {str(e)}"


def create_agent():
    llm = OllamaLLM(
        model="llama3.2",        # local model via Ollama
        temperature=0.1,         # low temperature for factual accuracy
        num_ctx=4096             # context window size
    )

    tools = [
        Tool(
            name="SearchDocuments",
            func=retrieve_relevant_chunks,
            description=(
                "Search through uploaded documents for relevant information. "
                "Use this FIRST for any question. Input should be a search query."
            )
        ),
        Tool(
            name="SearchWeb",
            func=search_web,
            description=(
                "Search the web for current information not available in documents. "
                "Use when documents do not contain the answer or when current "
                "information is needed. Input should be a search query."
            )
        )
    ]

    # Keep last 5 conversation exchanges in context
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=5,
        return_messages=True
    )

    # Pull the standard ReAct prompt from LangChain Hub
    prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,                # prints agent reasoning to terminal
        handle_parsing_errors=True,  # recovers from malformed LLM output
        max_iterations=5             # prevents infinite loops
    )
