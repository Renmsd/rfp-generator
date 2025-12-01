# orchestrator_runner.py
from langchain_openai import ChatOpenAI
from graph_core.orchestrator_graph import build_orchestrator_graph
import asyncio

llm = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0
)

graph = build_orchestrator_graph(llm)

async def run_orchestrator(raw_input: dict, include_sections: dict):

    initial_state = {
        "raw_input": raw_input,
        "decisions": {},
        "sections": [],
        "include_sections": include_sections,  # ❤️ this is the fix
        "completed_sections": []
    }

    # Graph is synchronous — call inside a thread
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, lambda: graph.invoke(initial_state)
    )

    return result.get("decisions", {})
