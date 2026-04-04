"""
LangChain Python Agent  —  agent.py
=====================================
Built strictly from: https://docs.langchain.com/oss/python/langchain/agents

Key APIs used (per docs):
  - create_agent()          from langchain.agents
  - @tool                   from langchain.tools
  - @wrap_model_call        from langchain.agents.middleware  (dynamic model)
  - @wrap_tool_call         from langchain.agents.middleware  (tool error handling)
  - agent.invoke()          standard LangGraph invocation
  - agent.stream()          streaming with stream_mode="values"

Run:  python agent.py
"""

from dotenv import load_dotenv
load_dotenv()   # reads .env → injects into os.environ

import os
import urllib.request
import json

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

# ── 1. Model ─────────────────────────────────────────────────────────────────
# Per docs: pass a model instance for full control over config
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,        # deterministic — best for agents
    max_tokens=1000,
    timeout=30,
    api_key=os.environ["OPENAI_API_KEY"],
)

# ── 2. Tools  (@tool decorator — per docs) ────────────────────────────────────
# Per docs: "Tools can be specified as plain Python functions or coroutines.
#  The @tool decorator can be used to customize tool names, descriptions,
#  argument schemas, and other properties."

@tool
def web_search(query: str) -> str:
    """Search the web for current events, recent news, or any information
    you don't already know. Input should be a concise search query string."""
    url = (
        f"https://api.duckduckgo.com/"
        f"?q={urllib.parse.quote(query)}&format=json&no_redirect=1"
    )
    with urllib.request.urlopen(url, timeout=8) as resp:
        data = json.loads(resp.read())
    answer = (
        data.get("AbstractText")
        or data.get("Answer")
        or (data.get("RelatedTopics") or [{}])[0].get("Text", "")
    )
    return (
        f'Search result for "{query}":\n{answer}'
        if answer
        else f'No instant answer found for "{query}". Try a more specific query.'
    )


@tool
def wikipedia_lookup(topic: str) -> str:
    """Look up encyclopaedic facts, historical events, scientific concepts,
    or any topic well-covered by Wikipedia. Input should be a topic name."""
    url = (
        f"https://en.wikipedia.org/api/rest_v1/page/summary/"
        f"{urllib.parse.quote(topic)}"
    )
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read())
        return f"Wikipedia — {data['title']}:\n{data['extract']}"
    except Exception:
        return f'No Wikipedia article found for "{topic}".'


@tool
def math_evaluator(expression: str) -> str:
    """Evaluate a safe mathematical Python expression.
    Supports: +, -, *, /, **, sqrt(), abs(), round(), pi, etc.
    Example: 'math.sqrt(144) * math.pi'"""
    import math   # noqa: PLC0415
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed_names["abs"] = abs
    allowed_names["round"] = round
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return f"Result of `{expression}` = {result}"
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


# ── 3. Tool error-handling middleware (per docs @wrap_tool_call) ──────────────
@wrap_tool_call
def handle_tool_errors(request, handler):
    """Catch any tool exception and return a ToolMessage instead of crashing."""
    try:
        return handler(request)
    except Exception as exc:
        return ToolMessage(
            content=f"Tool error — please check your input and try again. ({exc})",
            tool_call_id=request.tool_call["id"],
        )


# ── 4. Agent (create_agent — per docs) ───────────────────────────────────────
# Per docs: create_agent(model, tools=..., system_prompt=..., name=...)
agent = create_agent(
    model,
    tools=[web_search, wikipedia_lookup, math_evaluator],
    system_prompt=(
        "You are a helpful research assistant. "
        "Use tools to fetch accurate, current information. "
        "Be concise and cite your sources."
    ),
    middleware=[handle_tool_errors],
    name="research_agent",   # snake_case as recommended by docs
)

# ── 5. Streaming invocation (per docs agent.stream with stream_mode="values") ─
def run_query(query: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"QUERY: {query}")
    print("─" * 60)

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        latest = chunk["messages"][-1]
        if isinstance(latest, AIMessage):
            if latest.content:
                print(f"Agent: {latest.content}")
            elif latest.tool_calls:
                names = [tc["name"] for tc in latest.tool_calls]
                print(f"  → calling tools: {', '.join(names)}")


# ── 6. Demo ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import urllib.parse  # noqa: PLC0415  (needed inside tool at runtime)

    queries = [
        # maths via math_evaluator
        "What is math.sqrt(144) multiplied by math.pi? Show the calculation.",
        # current info via web search
        "What is the latest stable release of Python?",
        # encyclopaedic lookup via Wikipedia
        "Give me a brief summary of the James Webb Space Telescope.",
    ]

    print("═" * 60)
    print("  LangChain Python Agent  —  create_agent demo")
    print("═" * 60)

    for q in queries:
        run_query(q)