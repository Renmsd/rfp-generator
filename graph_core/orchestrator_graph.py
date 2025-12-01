# nodes/orchestrator_graph.py
from langgraph.graph import StateGraph, START, END
from datetime import datetime, timedelta
from typing import TypedDict, Annotated
from graph_core.field_map import FIELD_MAP
import operator
import asyncio


# -----------------------------------------------------
# State Format
# -----------------------------------------------------
class State(TypedDict):
    raw_input: str
    decisions: dict
    sections: list[str]
    include_sections: dict
    completed_sections: Annotated[list, operator.add]


# -----------------------------------------------------
# Generate Auto Dates
# -----------------------------------------------------
def generate_auto_dates(issue_date: str | None):
    if issue_date:
        base = datetime.strptime(issue_date, "%Y-%m-%d")
    else:
        base = datetime.today()

    return {
        "Issue_Date": base.strftime("%Y-%m-%d"),
        "Participation_Confirmation_Letter": (base + timedelta(days=2)).strftime("%Y-%m-%d"),
        "Submission_of_Questions_and_Inquiries": (base + timedelta(days=5)).strftime("%Y-%m-%d"),
        "Submission_of_Proposals": (base + timedelta(days=10)).strftime("%Y-%m-%d"),
        "Opening_of_Proposals": (base + timedelta(days=11)).strftime("%Y-%m-%d"),
        "Award_Decision_Date": (base + timedelta(days=17)).strftime("%Y-%m-%d"),
        "Commencement_of_Work": (base + timedelta(days=30)).strftime("%Y-%m-%d"),
    }


# -----------------------------------------------------
# PURE ASYNC LLM CALL — SAFE FOR RAILWAY
# -----------------------------------------------------
async def _call_llm_async(llm, prompt: str) -> str:
    try:
        res = await llm.ainvoke(prompt)
        return getattr(res, "content", res).strip()
    except Exception as e:
        return f"LLM Error: {str(e)}"


# -----------------------------------------------------
# Orchestrator (build decisions + choose sections)
# -----------------------------------------------------
def orchestrator(state: State):

    state.setdefault("decisions", {})
    decisions = state["decisions"]

    raw = state.get("raw_input")

    if isinstance(raw, dict):
        decisions.update(raw)
    elif isinstance(raw, str):
        try:
            import json
            decisions.update(json.loads(raw))
        except Exception:
            pass

    # default penalty fields
    for k in ["Penalty_Deduction", "Penalty_Execute_On_Vendor", "Penalty_Suspend", "Penalty_Termination"]:
        decisions.setdefault(k, "")

    decisions.update(generate_auto_dates(decisions.get("Issue_Date")))

    include = state.get("include_sections", {})

    sections = []
    for key, v in FIELD_MAP.items():
        if v == "llm":
            if key in include and not include[key]:
                continue
            sections.append(key)

    decisions["raw_input"] = raw

    return {"sections": sections, "decisions": decisions}


# -----------------------------------------------------
# Generate LLM sections in parallel
# -----------------------------------------------------
async def generate_sections_async(llm, prompts, sections, d):
    completed = {}

    independent = [
        s for s in sections
        if s in prompts and s != "Bid_Evaluation_Criteria"
    ]

    async def _generate_parallel():
        tasks = []
        for sec in independent:
            prompt = prompts[sec].format(**d)
            tasks.append(_call_llm_async(llm, prompt))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for sec, res in zip(independent, results):
            completed[sec] = (
                res if isinstance(res, str)
                else "تعذر توليد النص."
            )

        d.update(completed)

        # Special section: Bid Evaluation
        if "Bid_Evaluation_Criteria" in sections:
            tech = d.get("Technical_Proposal_Documents", "")
            fin = d.get("Financial_Proposal_Documents", "")

            eval_prompt = f"""
تحليل المحتوى التالي لاستخراج عناصر التقييم:

العرض الفني:
{tech}

العرض المالي:
{fin}

المطلوب:

إنشاء نموذج "معايير تقييم العروض" جاهز للإدراج في كراسة الشروط.

التوجيهات:

أولا تقسيم المعايير إلى مستويين فقط:
- المستوى الأول: تقييم فني
- المستوى الثاني: تقييم مالي

ثانيا استخراج عناصر التقييم من محتوى العرض الفني والمالي أعلاه، وليس من خيالك.
لا تتجاوز خمسة عناصر فنية وعنصرين ماليين.

ثالثا توزيع النقاط يتم حسب طريقة الترسية الموضحة في الإدخال Award_Method:{d.get("Award_Method")}


- إذا كانت الترسية تعتمد على أفضل عرض فني فقط Best Technical Offer فليكن التركيز الأكبر للنقاط في الجانب الفني مع حصة بسيطة للمالي
- إذا كانت Best Value فيجب توزيع النقاط بشكل متوازن بين الفني والمالي
- إذا كانت Lowest Price فيكون الجانب المالي هو الأعلى وزنا ويكون الفني داعما

رابعا إخراج النتيجة في جدول فقط يحتوي الأعمدة:
المستوى الأول | المستوى الثاني | الوزن | النقاط

خامسا يمنع كتابة شرح أو فقرات أو تعريفات. الجدول فقط.

ثامنا مهم جدا:
يمنع استخدام الأقواس بجميع أنواعها سواء كانت دائرية أو مربعة أو معقوفة.
اكتب النص بدون أي أقواس.

أخيرا اختم بجملة رسمية:
يتم ترسية المنافسة على العرض الحاصل على أعلى مجموع نقاط بعد التقييم الفني والمالي.
"""

            d["Bid_Evaluation_Criteria"] = await _call_llm_async(llm, eval_prompt)

        return d

    return await _generate_parallel()


# -----------------------------------------------------
# Sync wrapper for LangGraph (used inside FastAPI)
# -----------------------------------------------------
def generate_all_sections(state, llm):
    from graph_core.prompts import PROMPTS

    d = state["decisions"]
    sections = state["sections"]

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    new_decisions = loop.run_until_complete(
        generate_sections_async(llm, PROMPTS, sections, d)
    )
    loop.close()

    return {"decisions": new_decisions}


# -----------------------------------------------------
# Synthesizer
# -----------------------------------------------------
def synthesizer(state):
    return {"decisions": state["decisions"]}


# -----------------------------------------------------
# Build Graph
# -----------------------------------------------------
def build_orchestrator_graph(llm):
    g = StateGraph(State)

    g.add_node("orchestrator", orchestrator)
    g.add_node("generate_all_sections", lambda s: generate_all_sections(s, llm))
    g.add_node("synthesizer", synthesizer)

    g.add_edge(START, "orchestrator")
    g.add_edge("orchestrator", "generate_all_sections")
    g.add_edge("generate_all_sections", "synthesizer")
    g.add_edge("synthesizer", END)

    return g.compile()
