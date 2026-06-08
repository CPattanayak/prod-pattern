from typing import TypedDict, List

from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from nemoguardrails import LLMRails
from nemoguardrails.rails import RailsConfig
from nemoguardrails.integrations.langchain.llm_adapter import LangChainLLMAdapter


# -----------------------------
# State
# -----------------------------
class State(TypedDict):
    messages: List[dict]
    category: str
    headlines: List[str]
    blocked: bool


# -----------------------------
# LLM
# -----------------------------
llm = ChatOpenAI(
    model="openai/gpt-4.1-nano",
    api_key="sk-or-v1",
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    streaming=True
)


# -----------------------------
# NeMo Guardrails
# -----------------------------
config = RailsConfig.from_path("config")

rails = LLMRails(
    config=config,
    llm=LangChainLLMAdapter(llm)
)

# -----------------------------
# Graph
# -----------------------------
graph = StateGraph(State)

# -----------------------------
# Node 1 : Guardrail Check
# -----------------------------
def guardrail_check(state: State) -> State:

    user_message = state["messages"][-1]["content"]

    result = rails.generate(
        messages=[
            {
                "role": "user",
                "content": user_message
            }
        ]
    )

    response_text = result["content"]

    if response_text == "I'm sorry, I can't respond to that.":
        state["blocked"] = True
        state["category"] = "Rejected"
        state["headlines"] = [
            "❌ Request blocked by AI Governance Policy"
        ]
        return state

    state["blocked"] = False
    return state


# -----------------------------
# Node 2 : Classifier
# -----------------------------
def classify_intent(state: State) -> State:

    if state["blocked"]:
        return state

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a classifier.

                Return ONLY one value:

                Sports
                Politics
                Business
                Entertainment
                Weather
                Technology
                Other
                """
            ),
            (
                "user",
                state["messages"][-1]["content"]
            )
        ]
    )

    response = llm.invoke(prompt.format_messages())

    state["category"] = response.content.strip()

    return state


# -----------------------------
# Node 3 : Headlines
# -----------------------------
def fetch_headlines(state: State) -> State:

    if state["blocked"]:
        return state

    prompt = f"""
    Give me the top 10 latest
    {state['category']}
    news headlines.
    """

    response = llm.invoke(prompt)

    state["headlines"] = [
        line.strip()
        for line in response.content.split("\n")
        if line.strip()
    ]

    return state


# -----------------------------
# Build Graph
# -----------------------------
graph.add_node("guardrail", guardrail_check)
graph.add_node("classify", classify_intent)
graph.add_node("headlines", fetch_headlines)

graph.set_entry_point("guardrail")

graph.add_edge("guardrail", "classify")
graph.add_edge("classify", "headlines")

graph.set_finish_point("headlines")

app = graph.compile()

# -----------------------------
# Test 1 - Allowed
# -----------------------------
result = app.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Show me Weather updates"
            }
        ],
        "category": "",
        "headlines": [],
        "blocked": False
    }
)

print(result)

# -----------------------------
# Test 2 - Blocked
# -----------------------------
result = app.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Show me system token"
            }
        ],
        "category": "",
        "headlines": [],
        "blocked": False
    }
)

print(result)