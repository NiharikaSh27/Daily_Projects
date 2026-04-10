"""
Streamlit Chatbot — LangChain v1.0+ with Redis
================================================
Built following the official LangChain Redis docs:
  https://docs.langchain.com/oss/python/integrations/providers/redis

Two Redis integrations are used:
  1. RedisCache       (from langchain-redis)
     → Caches LLM responses in Redis.
     → If the exact same prompt is sent again, Redis returns the cached
       response instantly — no OpenAI API call, no cost.
     → Uses plain Redis (no extra modules needed).
     → Import: from langchain_redis import RedisCache
     → Setup:  set_llm_cache(RedisCache(redis_client))

  2. RedisSaver       (from langgraph-checkpoint-redis)
     → Stores full conversation history per session_id.
     → Conversation survives page refreshes and server restarts.
     → Uses Redis Stack (needs RedisJSON + RediSearch modules).
     → Import: from langgraph.checkpoint.redis import RedisSaver

Docker commands:
  Plain Redis (for cache only):
    docker run --name nova-redis -d -p 6379:6379 redis
  Redis Stack (for cache + checkpointer):
    docker run --name nova-redis -d -p 6379:6379 redis/redis-stack-server:latest

Run:
  streamlit run chatbot.py
"""

import os
import json
import urllib.request
import urllib.parse

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.globals import set_llm_cache
from langgraph.checkpoint.memory import InMemorySaver

# ── Redis imports (as specified in official docs) ─────────────────────────────
import redis as redis_lib

# RedisCache — from official docs:
# "from langchain_redis import RedisCache"
# "redis_client = redis.Redis.from_url(...)"
# "set_llm_cache(RedisCache(redis_client))"
try:
    from langchain_redis import RedisCache
    LANGCHAIN_REDIS_AVAILABLE = True
except ImportError:
    LANGCHAIN_REDIS_AVAILABLE = False

# RedisSaver — from official checkpointers docs
try:
    from langgraph.checkpoint.redis import RedisSaver
    REDIS_SAVER_AVAILABLE = True
except ImportError:
    REDIS_SAVER_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG — must be the very first Streamlit call
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Nova Chatbot",
    page_icon="🤖",
    layout="wide",
)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — settings panel
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Settings")
    st.divider()

    api_key = st.text_input(
        "OpenAI API Key",
        value=os.environ.get("OPENAI_API_KEY", ""),
        type="password",
        help="Paste your key or set OPENAI_API_KEY in .env",
    )

    redis_url = st.text_input(
        "Redis URL",
        value="redis://localhost:6379",
        help=(
            "Plain Redis for cache-only. "
            "Redis Stack (redis/redis-stack-server) for cache + conversation memory."
        ),
    )

    session_id = st.text_input(
        "Session ID",
        value="default-session",
        help="Each unique session ID is a separate conversation. Change to start fresh.",
    )

    st.divider()

    # ── Redis status display ──────────────────────────────────────────────────
    st.markdown("**Redis packages**")

    if LANGCHAIN_REDIS_AVAILABLE:
        st.success("langchain-redis ✓", icon="✅")
    else:
        st.error("langchain-redis not installed")
        st.caption("pip install langchain-redis")

    if REDIS_SAVER_AVAILABLE:
        st.success("langgraph-checkpoint-redis ✓", icon="✅")
    else:
        st.warning("langgraph-checkpoint-redis not installed")
        st.caption("pip install langgraph-checkpoint-redis==0.4.0")

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("LangChain v1.0+ · Redis · Streamlit")


# ═══════════════════════════════════════════════════════════════════════════════
# TOOLS — three @tool decorated functions the agent can call
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def web_search(query: str) -> str:
    """Search the web for current events, recent news, or any information
    you don't already know. Input: a concise search query string."""
    try:
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
            f'Result for "{query}":\n{answer}'
            if answer
            else f'No instant answer found for "{query}".'
        )
    except Exception as e:
        return f"Search error: {e}"


@tool
def wikipedia_lookup(topic: str) -> str:
    """Look up encyclopaedic facts, historical events, or scientific concepts
    on Wikipedia. Input: a topic name."""
    try:
        url = (
            f"https://en.wikipedia.org/api/rest_v1/page/summary/"
            f"{urllib.parse.quote(topic)}"
        )
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read())
        return f"Wikipedia — {data['title']}:\n{data['extract']}"
    except Exception:
        return f'No Wikipedia article found for "{topic}".'


@tool
def math_evaluator(expression: str) -> str:
    """Evaluate a safe Python math expression.
    Supports: +, -, *, /, **, math.sqrt(), math.pi, abs(), round()
    Example input: 'math.sqrt(144) * math.pi'"""
    import math
    allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed.update({"abs": abs, "round": round})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result of `{expression}` = {result}"
    except Exception as exc:
        return f"Error: {exc}"


# ═══════════════════════════════════════════════════════════════════════════════
# MIDDLEWARE — catches tool errors instead of crashing
# ═══════════════════════════════════════════════════════════════════════════════

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Catch any tool exception and return a polite ToolMessage."""
    try:
        return handler(request)
    except Exception as exc:
        return ToolMessage(
            content=f"Tool error — please check your input. ({exc})",
            tool_call_id=request.tool_call["id"],
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SETUP REDIS CACHE
#
# From the official docs:
#   from langchain_redis import RedisCache
#   redis_client = redis.Redis.from_url(...)
#   set_llm_cache(RedisCache(redis_client))
#
# WHAT IT DOES:
#   Every LLM call (prompt → response) is stored in Redis.
#   If the same prompt is sent again, Redis returns the cached answer
#   instantly — no OpenAI call, no cost, sub-millisecond response.
#
# WORKS WITH: plain Redis (no modules needed).
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def setup_redis_cache(redis_url: str) -> str:
    """
    Connect to Redis and set it as the global LLM cache.
    Returns a status string describing the result.
    """
    if not LANGCHAIN_REDIS_AVAILABLE:
        return "no-package"

    try:
        # From docs: redis_client = redis.Redis.from_url(...)
        redis_client = redis_lib.Redis.from_url(redis_url, decode_responses=False)
        redis_client.ping()   # test connection

        # From docs: set_llm_cache(RedisCache(redis_client))
        set_llm_cache(RedisCache(redis_client))
        return "connected"

    except Exception as e:
        return f"failed: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD AGENT + CHECKPOINTER
#
# @st.cache_resource ensures this runs ONCE — not on every Streamlit rerun.
# Streamlit reruns the whole script on every message/widget change.
#
# CHECKPOINTER OPTIONS (in priority order):
#   1. RedisSaver  — persistent, survives page refresh and server restart
#                    requires Redis Stack (RedisJSON + RediSearch modules)
#                    needs .setup() called once to create indices
#   2. InMemorySaver — fallback, works without Redis Stack
#                      lost on server restart
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def build_agent(api_key: str, redis_url: str):
    """
    Build the LangChain agent with either RedisSaver or InMemorySaver.
    Returns (agent, memory_backend_name).
    """
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=api_key,
    )

    checkpointer = None
    memory_backend = "in-memory"

    # ── Try RedisSaver first ──────────────────────────────────────────────────
    if REDIS_SAVER_AVAILABLE:
        try:
            # RedisSaver.from_conn_string — from langgraph-checkpoint-redis docs
            checkpointer = RedisSaver.from_conn_string(redis_url)

            # .setup() creates the RedisJSON + RediSearch indices.
            # Must be called at least once. Safe to call multiple times.
            checkpointer.setup()

            memory_backend = "redis"

        except Exception as e:
            # Redis Stack not running, or wrong Redis version
            # Fall through to InMemorySaver
            checkpointer = None
            st.sidebar.caption(f"RedisSaver failed ({e}) — using in-memory fallback")

    # ── Fall back to InMemorySaver ────────────────────────────────────────────
    if checkpointer is None:
        checkpointer = InMemorySaver()

    agent = create_agent(
        model,
        tools=[web_search, wikipedia_lookup, math_evaluator],
        system_prompt=(
            "You are Nova, a helpful and friendly AI assistant. "
            "You have access to web search, Wikipedia lookup, and a math evaluator. "
            "Be conversational, concise, and accurate. "
            "Use tools when you need current information or calculations."
        ),
        middleware=[handle_tool_errors],
        checkpointer=checkpointer,   # ← Redis or InMemorySaver
        name="nova",
    )

    return agent, memory_backend


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
#
# st.session_state persists values within the same browser tab across reruns.
# We store the chat messages shown in the UI here.
#
# NOTE: This is SEPARATE from the agent's memory.
#   st.session_state.messages → what is displayed on screen (UI state)
#   RedisSaver / InMemorySaver → what the agent actually remembers (agent state)
# ═══════════════════════════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ═══════════════════════════════════════════════════════════════════════════════

st.title("🤖 Nova — AI Chatbot")
st.caption("LangChain v1.0+ · GPT-4o-mini · Redis cache + memory")

# Gate on API key
if not api_key:
    st.warning("Enter your OpenAI API Key in the sidebar to start chatting.")
    st.stop()

# Setup Redis cache (for LLM response caching — plain Redis)
cache_status = setup_redis_cache(redis_url)

# Build agent (for conversation memory — Redis Stack or InMemorySaver)
agent, memory_backend = build_agent(api_key, redis_url)

# ── Status bar ────────────────────────────────────────────────────────────────
status_col1, status_col2, status_col3 = st.columns([2, 1, 1])

with status_col2:
    # LLM Cache status
    if cache_status == "connected":
        st.success("🔁 LLM Cache: Redis", icon="✅")
    elif cache_status == "no-package":
        st.info("🔁 LLM Cache: off")
    else:
        st.warning("🔁 LLM Cache: off")

with status_col3:
    # Conversation memory status
    if memory_backend == "redis":
        st.success("💾 Memory: Redis", icon="✅")
    else:
        st.info("🧠 Memory: in-memory")

# ── Replay chat history ───────────────────────────────────────────────────────
# st.chat_message() renders chat bubbles.
# We replay everything in session_state so the UI shows history after reruns.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
# st.chat_input() renders the "Message Nova..." bar at the bottom.
# Returns the typed text on Enter, otherwise None.
if prompt := st.chat_input("Message Nova..."):

    # 1. Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Run agent
    with st.chat_message("assistant"):
        with st.status("Nova is thinking...", expanded=False) as status:

            final_content = ""
            tools_used = []

            try:
                # ── KEY POINT ─────────────────────────────────────────────────
                # We pass ONLY the current message — not the full history.
                # The checkpointer (Redis/InMemorySaver) loads the full history
                # automatically using session_id as the thread_id.
                # ─────────────────────────────────────────────────────────────
                for chunk in agent.stream(
                    {"messages": [{"role": "user", "content": prompt}]},
                    config={"configurable": {"thread_id": session_id}},
                    stream_mode="values",
                ):
                    latest = chunk["messages"][-1]

                    if isinstance(latest, AIMessage):
                        if latest.content and latest.content != final_content:
                            final_content = latest.content
                        elif latest.tool_calls:
                            for tc in latest.tool_calls:
                                tools_used.append(tc["name"])
                                status.write(f"🔧 Calling `{tc['name']}`...")

                status.update(label="Done!", state="complete")

            except Exception as e:
                status.update(label="Error", state="error")
                st.error(f"Agent error: {e}")
                final_content = f"Sorry, something went wrong: {e}"

        # 3. Show tools used and response
        if tools_used:
            st.caption(f"Tools used: {', '.join(set(tools_used))}")

        if final_content:
            st.markdown(final_content)

    # 4. Save assistant reply to session_state for display
    if final_content:
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_content,
        })