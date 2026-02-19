"""
Pydantic 데이터 모델 — 세금계산서 발행 데이터 세팅용
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


# ── 요청/응답 ──

class ChatRequest(BaseModel):
    """POST /chat/{user_id} 요청 바디"""
    question: str


class ChatResponse(BaseModel):
    """POST /chat/{user_id} 응답 바디"""
    type: str                          # "message" | "data"
    text: str                          # 사용자에게 보여줄 메시지
    result: ChatResult | None = None


class ChatResult(BaseModel):
    """응답 내부 결과"""
    action: str | None = None          # "ready_invoice" | "need_more_info" | "need_client_choice" | "ask_company"
    raw: dict[str, Any] | None = None


# ── 세금계산서 슬롯 ──

class InvoiceSlots(BaseModel):
    """세금계산서 필수 정보"""
    seq: str | None = None       # 사업자코드
    company: str | None = None   # 거래처명
    item: str | None = None      # 품목
    amount: str | None = None    # 금액 (숫자 문자열)
    date: str | None = None      # 날짜 (YYYY-MM-DD)

    def missing_fields(self) -> list[str]:
        labels = {"company": "거래처", "item": "품목", "amount": "금액"}
        return [label for key, label in labels.items() if not getattr(self, key)]

    def is_complete(self) -> bool:
        return len(self.missing_fields()) == 0

    def merge(self, other: InvoiceSlots) -> InvoiceSlots:
        """other의 non-null 값으로 현재 슬롯을 덮어쓴다."""
        return InvoiceSlots(
            seq=other.seq or self.seq,
            company=other.company or self.company,
            item=other.item or self.item,
            amount=other.amount or self.amount,
            date=other.date or self.date,
        )

    def summary(self) -> str:
        def _or(v: str | None) -> str:
            return v if v else "(없음)"
        return f"거래처={_or(self.company)}, 품목={_or(self.item)}, 금액={_or(self.amount)}, 날짜={_or(self.date)}"


# ── 거래처 후보 (RAG) ──

class CompanyCandidate(BaseModel):
    seq: str
    company: str
    score: float = 0.0


class MatchStatus(str, Enum):
    EXACT_MATCH = "exact_match"
    MULTIPLE_MATCHES = "multiple_matches"
    NO_MATCH = "no_match"


class CompanySearchResult(BaseModel):
    status: MatchStatus
    candidates: list[CompanyCandidate] = []

    def best_match(self) -> CompanyCandidate | None:
        return self.candidates[0] if self.candidates else None


# ── 대화 이력 ──

class MessageRow(BaseModel):
    role: str        # "user" | "assistant"
    content: str


# ── 의도 분류 ──

class UserIntent(str, Enum):
    ISSUE_INVOICE = "ISSUE_INVOICE"
    HISTORY_BASED = "HISTORY_BASED"
    CANCEL = "CANCEL"
    OTHER = "OTHER"
