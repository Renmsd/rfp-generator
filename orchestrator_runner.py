from langchain_openai import ChatOpenAI
from graph_core.orchestrator_graph import build_orchestrator_graph
import asyncio
import os

llm = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0
)

graph = build_orchestrator_graph(llm)

async def run_orchestrator(raw_input: dict, include_sections: dict):
    # 🤍 Inject include_sections into state (not Flask)
    initial_state = {
        "raw_input": raw_input,
        "include_sections": include_sections,   # <-- important
        "decisions": {},
        "sections": [],
        "completed_sections": []
    }

    result = graph.invoke(initial_state)
    return result.get("decisions", {})
