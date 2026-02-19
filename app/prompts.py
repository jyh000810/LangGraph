"""
프롬프트 템플릿 — 세금계산서 발행 데이터 세팅용
"""

from __future__ import annotations

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage

from app.models import InvoiceSlots


# ── IntentRouter 시스템 프롬프트 ──

INTENT_ROUTER_SYSTEM = """\
세금계산서 발행 데이터 세팅 라우터입니다.
사용자의 현재 의도를 아래 분류 중 정확히 하나로 판단하세요.

[현재 상태]
{current_state}

[분류 기준]
- ISSUE_INVOICE: 신규 발행 요청 또는 거래처/품목/금액 입력 (수정 포함)
- HISTORY_BASED: "방금", "이전", "아까", "다시", "같은 거", "저번" 등 과거 이력 재사용
- CANCEL: "취소", "그만", "처음부터", "중단" 등 작업 중단
- OTHER: 위 어디에도 해당하지 않는 일반 대화

[판단 지침]
1. 현재 상태가 '세금계산서 데이터 수집 중' → ISSUE_INVOICE
   - 단, CANCEL·HISTORY_BASED 의도가 명확하면 해당 분류 우선

2. 현재 상태가 '새 요청 대기 중':
   - 거래처/품목/금액 중 하나라도 포함 → ISSUE_INVOICE

반드시 ISSUE_INVOICE, HISTORY_BASED, CANCEL, OTHER 중 하나만 대답하세요.

대화 이력:
{chat_history}
"""

# ── SlotExtractor 시스템 프롬프트 ──

SLOT_EXTRACTOR_SYSTEM = """\
당신은 세금계산서 데이터 추출기입니다.
사용자 입력에서 거래처(company), 품목(item), 금액(amount), 날짜(date)를 추출하세요.
확인할 수 없는 필드는 null로 남겨두세요.

**금액 규칙:**
- 금액은 사용자가 말한 표현 그대로 추출하세요. 숫자로 변환하지 마세요.
- 단, 명백한 오타만 보정하세요.
- 예시: "오천만원" → "오천만원" (그대로)
- 예시: "500만원" → "500만원" (그대로)
- 예시: "1억5천만" → "1억5천만" (그대로)
- 예시: "오쳔만원" → "오천만원" (오타 보정)

**날짜 규칙:**
- 날짜는 반드시 YYYY-MM-DD 형식으로 변환하세요.
- 오늘 날짜: {today}
- "오늘" → 오늘 날짜, "내일" → 내일 날짜, "모레" → 모레 날짜
- "어제" → 어제 날짜, "12월 25일" → 올해 12월 25일
- 날짜 언급이 없으면 오늘 날짜로 추출하세요.

반드시 아래 JSON 형식으로만 응답하세요 (다른 텍스트 없이):
{{"company": "...", "item": "...", "amount": "...", "date": "..."}}

기존 입력 데이터: {existing_slots}

대화 이력:
{chat_history}
"""

# ── InvoiceSelector 시스템 프롬프트 ──

INVOICE_SELECTOR_SYSTEM = """\
사용자가 과거 발행 데이터를 재사용하려고 합니다.
아래 발행 내역을 참고하여 몇 번째 전 데이터를 원하는지 판단하세요.

발행 내역 (최신순):
{invoice_summary}

index 규칙: 0=가장 최근, 1=두 번째 전, 2=세 번째 전...
명시적 표현이 없으면 0(가장 최근)을 반환하세요.
숫자만 반환하세요.
"""

# ── OTHER 프롬프트 ──

OTHER_INSTRUCTION = """\
당신은 세금계산서 발행 도우미 챗봇입니다.
사용자와 자연스럽게 대화하되, 대화 마지막에 세금계산서 발행 안내를 추가하세요.
예: "세금계산서 발행이 필요하시면 거래처명, 품목, 금액을 알려주세요!"
"""

OTHER_FALLBACK = "무엇을 도와드릴까요? 세금계산서 발행을 원하시면 거래처/품목/금액을 알려주세요."


# ── 프롬프트 빌드 헬퍼 함수들 ──

def build_current_state_from_slots(slots: InvoiceSlots | None, status: str) -> str:
    """현재 상태를 IntentRouter에 전달할 한 줄 설명으로 변환."""
    if slots and (slots.company or slots.item or slots.amount) and status in ("extracting", ""):
        return "세금계산서 데이터 수집 중"
    return "새 요청 대기 중"


def build_history_text_from_messages(messages: list[AnyMessage], limit: int = 30) -> str:
    """AnyMessage 목록에서 LLM 전달용 역할별 문자열로 변환."""
    lines = []
    for msg in messages[-limit:]:
        if isinstance(msg, HumanMessage):
            lines.append(f"[user]: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"[assistant]: {msg.content}")
    return "\n".join(lines)


def build_invoice_summary(invoices: list[InvoiceSlots]) -> str:
    """과거 발행 이력을 LLM 전달용 요약 문자열로 변환."""
    lines = []
    for i, slots in enumerate(invoices):
        _or = lambda v: v if v else "(없음)"
        lines.append(
            f"[{i}번째 전] 거래처={_or(slots.company)}, 품목={_or(slots.item)}, "
            f"금액={_or(slots.amount)}, 날짜={_or(slots.date)}"
        )
    return "\n".join(lines)
