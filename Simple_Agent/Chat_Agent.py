"""
LangChain Python Chat Agent  —  chat_agent.py
===============================================
Built strictly from: https://docs.langchain.com/oss/python/langchain/agents

Extra features vs agent.py:
  - @dynamic_prompt middleware  → system prompt adapts to conversation length
  - AgentState + state_schema   → custom state (tracks user_preferences)
  - agent.stream()              → real-time streaming to stdout
  - Full conversation memory    → messages accumulate across turns

Run:  python chat_agent.py
"""

from dotenv import load_dotenv
load_dotenv()

import os
import json
import urllib.request
import urllib.parse
from typing import TypedDict

from langchain.agents import create_agent, AgentState
from langchain.tools import tool
from langchain.agents.middleware import (
    wrap_tool_call,
    dynamic_prompt,
    ModelRequest,
)
from langchain.messages import ToolMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

# ── Model ─────────────────────────────────────────────────────────────────────
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.environ["OPENAI_API_KEY"],
)

# ── Tools (@tool decorator — per docs) ────────────────────────────────────────
@tool
def web_search(query: str) -> str:
    """Search the web for current or recent information.
    Input should be a concise search query string."""
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
        else f'No instant answer found for "{query}".'
    )


@tool
def wikipedia_lookup(topic: str) -> str:
    """Look up encyclopaedic facts and concepts on Wikipedia.
    Input should be a topic name."""
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
    """Evaluate a safe Python math expression.
    Supports math.sqrt(), math.pow(), math.pi, +, -, *, /, ** etc."""
    import math
    allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed.update({"abs": abs, "round": round})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return f"Result of `{expression}` = {result}"
    except Exception as exc:
        return f"Error: {exc}"


# ── Tool error handler middleware (per docs @wrap_tool_call) ──────────────────
@wrap_tool_call
def handle_tool_errors(request, handler):
    try:
        return handler(request)
    except Exception as exc:
        return ToolMessage(
            content=f"Tool error — please check your input. ({exc})",
            tool_call_id=request.tool_call["id"],
        )


# ── Dynamic system prompt middleware (per docs @dynamic_prompt) ───────────────
# Per docs: "The @dynamic_prompt decorator creates middleware that generates
#  system prompts based on the model request"
@dynamic_prompt
def adaptive_system_prompt(request: ModelRequest) -> str:
    """Adapt the system prompt based on conversation length."""
    msg_count = len(request.state.get("messages", []))
    base = (
        "You are a helpful research assistant with access to web search, "
        "Wikipedia, and a math evaluator."
    )
    if msg_count > 10:
        return f"{base} The conversation is getting long — be extra concise."
    return f"{base} Be accurate and cite your sources."


# ── Custom state schema (per docs AgentState + TypedDict) ────────────────────
# Per docs: "Custom state schemas must extend AgentState as a TypedDict"
class ChatState(AgentState):
    user_preferences: dict   # tracks any preferences mentioned in chat


# ── Agent (create_agent with state_schema + middleware — per docs) ────────────
agent = create_agent(
    model,
    tools=[web_search, wikipedia_lookup, math_evaluator],
    state_schema=ChatState,
    middleware=[handle_tool_errors, adaptive_system_prompt],
    name="chat_research_agent",
)

# ── Streaming chat loop ───────────────────────────────────────────────────────
def chat():
    print("\n🤖  LangChain Chat Agent  (type 'quit' to exit)")
    print("    Streaming  |  Dynamic prompt  |  Custom state\n")

    # Build up messages across turns for memory
    messages: list = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        print("\nAgent: ", end="", flush=True)
        final_content = ""

        # Per docs: agent.stream() with stream_mode="values"
        for chunk in agent.stream(
            {
                "messages": messages,
                "user_preferences": {},   # custom state field
            },
            stream_mode="values",
        ):
            latest = chunk["messages"][-1]

            if isinstance(latest, AIMessage):
                if latest.content and latest.content != final_content:
                    # Print only the new delta
                    delta = latest.content[len(final_content):]
                    print(delta, end="", flush=True)
                    final_content = latest.content
                elif latest.tool_calls:
                    names = [tc["name"] for tc in latest.tool_calls]
                    print(f"\n  [using: {', '.join(names)}]", end="", flush=True)

        print("\n")

        # Append assistant reply to history for next turn
        if final_content:
            messages.append({"role": "assistant", "content": final_content})


if __name__ == "__main__":
    chat()