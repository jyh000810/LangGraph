"""
LangGraph 상태 정의 — Annotated reducer로 메시지 자동 누적
"""

from __future__ import annotations

from typing import Annotated, Any

from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from app.models import CompanyCandidate, InvoiceSlots


class InvoiceState(TypedDict, total=False):
    # ── 사용자/세션 ──
    user_id: str
    question: str

    # ── 메시지 이력 (add_messages로 자동 누적) ──
    messages: Annotated[list[AnyMessage], add_messages]

    # ── 슬롯 상태 ──
    slots: InvoiceSlots | None
    candidates: list[CompanyCandidate]

    # ── 라우팅 / 흐름 제어 ──
    intent: str    # UserIntent.value
    status: str    # "extracting" | "complete" | "cancelled" | "matched"
                   # "exact_match" | "multiple_matches" | "no_match" | "no_company" | "confirmed"

    # ── 최종 응답 (terminal 노드에서 채움) ──
    response_type: str              # "message" | "data"
    response_message: str
    response_action: str | None     # "ready_invoice" | "need_more_info" | "need_client_choice"
    response_data: dict[str, Any] | None
