import streamlit as st
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
# Streamlit UI
# -----------------------------
st.title("📰 LangGraph + NeMo Guardrails News App")

api_key = st.text_input("Enter your OpenRouter API Key:", type="password")

if api_key:
    # -----------------------------
    # LLMs
    # -----------------------------
    llm1 = ChatOpenAI(
        model="openai/gpt-4.1-nano",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        streaming=True
    )
    llm = ChatOpenAI(
        model="openrouter/auto",
        api_key=api_key,
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
        llm=LangChainLLMAdapter(llm1)
    )

    # -----------------------------
    # Graph
    # -----------------------------
    graph = StateGraph(State)

    def guardrail_check(state: State) -> State:
        user_message = state["messages"][-1]["content"]
        result = rails.generate(messages=[{"role": "user", "content": user_message}])
        response_text = result["content"]

        if response_text == "I'm sorry, I can't respond to that.":
            state["blocked"] = True
            state["category"] = "Rejected"
            state["headlines"] = ["❌ Request blocked by AI Governance Policy"]
            return state

        state["blocked"] = False
        return state

    def classify_intent(state: State) -> State:
        if state["blocked"]:
            return state

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a classifier.\nReturn ONLY one value:\nSports\nPolitics\nBusiness\nEntertainment\nWeather\nTechnology\nOther"),
            ("user", state["messages"][-1]["content"])
        ])
        response = llm.invoke(prompt.format_messages())
        state["category"] = response.content.strip()
        return state

    def fetch_headlines(state: State) -> State:
        if state["blocked"]:
            return state

        prompt = f"Give me the top 10 latest {state['category']} news headlines."
        response = llm.invoke(prompt)
        state["headlines"] = [line.strip() for line in response.content.split("\n") if line.strip()]
        return state

    graph.add_node("guardrail", guardrail_check)
    graph.add_node("classify", classify_intent)
    graph.add_node("headlines", fetch_headlines)

    graph.set_entry_point("guardrail")
    graph.add_edge("guardrail", "classify")
    graph.add_edge("classify", "headlines")
    graph.set_finish_point("headlines")

    app = graph.compile()

    # -----------------------------
    # User Input
    # -----------------------------
    user_input = st.text_input("Enter your query (e.g., 'Show me Weather updates')")

    if user_input:
        result = app.invoke({
            "messages": [{"role": "user", "content": user_input}],
            "category": "",
            "headlines": [],
            "blocked": False
        })

        if result["blocked"]:
            st.error("❌ Request blocked by AI Governance Policy")
            st.write("**Category:**", result["category"])
            st.write("**Headlines:**")
            for h in result["headlines"]:
                st.markdown(f"- {h}")
        else:
            st.success("✅ Request allowed")
            st.write("**Category:**", result["category"])
            st.write("**Headlines:**")
            for h in result["headlines"]:
                st.markdown(f"- {h}")
