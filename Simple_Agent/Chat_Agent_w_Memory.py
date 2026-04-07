"""
LangChain Chat Agent WITH Memory  —  chat_agent.py  (LangChain v1.0+)
=======================================================================
WHAT CHANGED:
  OLD: ConversationBufferMemory, ConversationBufferWindowMemory,
       ConversationSummaryMemory  — all removed from langchain.memory in v1.0

  NEW: InMemorySaver from langgraph.checkpoint.memory
       Single replacement for ALL three old memory types.
       The "window" and "summary" behaviour is now controlled separately
       using trim_messages (for window) or a summarization middleware.

SIMPLIFICATION:
  OLD chat loop: 4 steps per turn (load, append, stream, save)
  NEW chat loop: 1 step per turn (stream with thread_id) — rest is automatic

Run:  python chat_agent.py
"""

from dotenv import load_dotenv
load_dotenv()

import os, json, urllib.request, urllib.parse

from langchain.agents import create_agent, AgentState
from langchain.tools import tool
from langchain.agents.middleware import wrap_tool_call, dynamic_prompt, ModelRequest
from langchain.messages import ToolMessage, AIMessage

from langchain_openai import ChatOpenAI

# NEW memory import — langgraph, not langchain.memory
from langgraph.checkpoint.memory import InMemorySaver


# ── Model ─────────────────────────────────────────────────────────────────────
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.environ["OPENAI_API_KEY"],
)


# ── Tools (unchanged) ─────────────────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """Search the web for current or recent information.
    Input should be a concise search query string."""
    url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_redirect=1"
    with urllib.request.urlopen(url, timeout=8) as resp:
        data = json.loads(resp.read())
    answer = (
        data.get("AbstractText") or data.get("Answer")
        or (data.get("RelatedTopics") or [{}])[0].get("Text", "")
    )
    return f'Result for "{query}":\n{answer}' if answer else f'No result for "{query}".'


@tool
def wikipedia_lookup(topic: str) -> str:
    """Look up encyclopaedic facts and concepts on Wikipedia.
    Input should be a topic name."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(topic)}"
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
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result of `{expression}` = {result}"
    except Exception as exc:
        return f"Error: {exc}"


# ── Tool error handler (unchanged) ────────────────────────────────────────────

@wrap_tool_call
def handle_tool_errors(request, handler):
    try:
        return handler(request)
    except Exception as exc:
        return ToolMessage(
            content=f"Tool error — please check your input. ({exc})",
            tool_call_id=request.tool_call["id"],
        )


# ── Dynamic system prompt middleware (unchanged) ──────────────────────────────
# Reads message count from the agent's state and adapts the prompt

@dynamic_prompt
def adaptive_system_prompt(request: ModelRequest) -> str:
    """Adapt system prompt based on conversation length."""
    msg_count = len(request.state.get("messages", []))
    base = (
        "You are a helpful research assistant with access to web search, "
        "Wikipedia, and a math evaluator."
    )
    if msg_count > 10:
        return f"{base} The conversation is getting long — be extra concise."
    return f"{base} Be accurate and cite your sources."


# ── Custom state schema (unchanged) ───────────────────────────────────────────

class ChatState(AgentState):
    user_preferences: dict   # extra memory slot beyond messages


# ── Memory (NEW) ──────────────────────────────────────────────────────────────
#ss
# Replaces ALL three old classes in one object:
#   ConversationBufferMemory        -> InMemorySaver (stores full history)
#   ConversationBufferWindowMemory  -> InMemorySaver + trim_messages (advanced)
#   ConversationSummaryMemory       -> InMemorySaver + summary middleware (advanced)
#
# For most use cases InMemorySaver alone is all you need.

checkpointer = InMemorySaver()


# ── Agent (one new line: checkpointer=checkpointer) ───────────────────────────

agent = create_agent(
    model,
    tools=[web_search, wikipedia_lookup, math_evaluator],
    state_schema=ChatState,
    middleware=[handle_tool_errors, adaptive_system_prompt],
    checkpointer=checkpointer,   # <- memory lives here now
    name="chat_research_agent",
)


# ── Chat loop (massively simplified) ──────────────────────────────────────────
#
# OLD chat loop per turn (4 manual steps):
#   messages = memory_to_messages(memory)          # LOAD from memory
#   messages.append({"role":"user","content":...}) # append current question
#   agent.stream({"messages": messages})           # run agent
#   memory.save_context({"input":...},{"output":...}) # SAVE to memory
#
# NEW chat loop per turn (just stream with a thread_id):
#   agent.stream(
#       {"messages": [current_question]},
#       config={"configurable": {"thread_id": "session-1"}},
#   )
#   LangGraph loads history automatically before running.
#   LangGraph saves the new exchange automatically after running.
#   The helper function get_messages_from_memory() is completely gone.
#
# THREAD ID = the session identifier.
# Every user/conversation gets its own thread_id.
# Same thread_id -> same conversation history (memory persists).
# Different thread_id -> fresh conversation (no prior context).

def chat():
    print("\n" + "=" * 58)
    print("  LangChain Chat Agent  —  with InMemorySaver (v1.0+)")
    print("  Commands: 'quit' = exit")
    print("=" * 58 + "\n")

    # One thread_id per session = one conversation's memory
    # Change this string to start a completely fresh conversation
    SESSION_ID = "chat-session-1"

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Agent: Goodbye!")
            break

        # Pass ONLY the current message.
        # LangGraph reads all previous turns automatically using SESSION_ID.
        # No helper function needed. No manual list building.
        print("\nAgent: ", end="", flush=True)
        final_content = ""

        for chunk in agent.stream(
            {
                "messages": [{"role": "user", "content": user_input}],
                "user_preferences": {},   # custom state field from ChatState
            },
            config={"configurable": {"thread_id": SESSION_ID}},
            stream_mode="values",
        ):
            latest = chunk["messages"][-1]
            if isinstance(latest, AIMessage):
                if latest.content and latest.content != final_content:
                    delta = latest.content[len(final_content):]
                    print(delta, end="", flush=True)
                    final_content = latest.content
                elif latest.tool_calls:
                    names = [tc["name"] for tc in latest.tool_calls]
                    print(f"\n  [using: {', '.join(names)}]", end="", flush=True)

        print("\n")
        # No save_context() call — LangGraph already saved this exchange
        # The next call with the same SESSION_ID will see this exchange in history


if __name__ == "__main__":
    chat()