"""
LangGraph 그래프 정의 — MemorySaver + interrupt() + 순환 엣지

흐름:
  START → route_intent → (의도 분기)
    ├─ issue_invoice → extract_slots → search_company → (결과 분기)
    │     ├─ exact_match / confirmed / no_company → check_slots
    │     └─ multiple_matches → company_choice ← interrupt()
    │           ├─ matched      → check_slots
    │           ├─ extracting   → extract_slots  ← (루프)
    │           └─ cancelled    → cancel → END
    │
    │   check_slots → (완성도 분기)
    │     ├─ complete    → finalize → END
    │     ├─ extracting  → extract_slots  ← (루프)
    │     └─ cancelled   → cancel → END
    │
    ├─ history_based → select_history → END
    ├─ cancel        → cancel        → END
    └─ other         → general_chat  → END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.nodes import (
    cancel,
    check_slots,
    company_choice,
    decide_after_check_slots,
    decide_after_company_choice,
    decide_after_search,
    decide_intent,
    extract_slots,
    finalize,
    general_chat,
    route_intent,
    search_company,
    select_history,
)
from app.state import InvoiceState


def build_graph() -> StateGraph:
    builder = StateGraph(InvoiceState)

    # ── 노드 등록 ──
    builder.add_node("route_intent",    route_intent)
    builder.add_node("extract_slots",   extract_slots)
    builder.add_node("search_company",  search_company)
    builder.add_node("company_choice",  company_choice)
    builder.add_node("check_slots",     check_slots)
    builder.add_node("finalize",        finalize)
    builder.add_node("select_history",  select_history)
    builder.add_node("cancel",          cancel)
    builder.add_node("general_chat",    general_chat)

    # ── 진입점 ──
    builder.add_edge(START, "route_intent")

    # ── route_intent 이후 분기 ──
    builder.add_conditional_edges(
        "route_intent",
        decide_intent,
        {
            "issue_invoice":   "extract_slots",
            "history_based":   "select_history",
            "cancel":          "cancel",
            "other":           "general_chat",
        },
    )

    # ── extract_slots → search_company ──
    builder.add_edge("extract_slots", "search_company")

    # ── search_company 이후 분기 ──
    builder.add_conditional_edges(
        "search_company",
        decide_after_search,
        {
            "check_slots":      "check_slots",
            "company_choice":   "company_choice",
        },
    )

    # ── company_choice 이후 분기 (루프 포함) ──
    builder.add_conditional_edges(
        "company_choice",
        decide_after_company_choice,
        {
            "check_slots":    "check_slots",
            "extract_slots":  "extract_slots",   # ← 루프
            "cancel":         "cancel",
        },
    )

    # ── check_slots 이후 분기 (루프 포함) ──
    builder.add_conditional_edges(
        "check_slots",
        decide_after_check_slots,
        {
            "finalize":       "finalize",
            "extract_slots":  "extract_slots",   # ← 루프
            "cancel":         "cancel",
        },
    )

    # ── 종료 엣지 ──
    for node in ["finalize", "select_history", "cancel", "general_chat"]:
        builder.add_edge(node, END)

    return builder
