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

반드시 아래 JSON 형식으로만 응답하세요:
{{"intent": "ISSUE_INVOICE" | "HISTORY_BASED" | "CANCEL" | "OTHER"}}

대화 이력:
{chat_history}
"""

# ── SlotExtractor 시스템 프롬프트 ──

SLOT_EXTRACTOR_SYSTEM = """\
당신은 세금계산서 발행 데이터 추출 전문가입니다.
반드시 [현재 메시지]에서만 정보를 추출하세요. 대화 이력은 사용하지 마세요.

### 추출 규칙:
1. **현재 메시지만 분석:** [현재 메시지]에 사용자가 직접 언급한 내용만 추출합니다.
   - 이전 대화 내용은 일절 사용하지 마세요.
   - [기존 입력 데이터]는 파이썬 시스템이 병합하므로 억지로 채우지 마세요.

2. **null 원칙:** [현재 메시지]에 없는 필드는 반드시 `null`로 반환하세요.

3. **수정 우선:** 현재 메시지의 정보가 [기존 입력 데이터]와 다르면 새 값으로 업데이트하세요.

4. **원문 보존:** 모든 필드는 사용자가 입력한 언어와 표현을 그대로 유지하세요.
   - 절대 다른 언어로 번역하거나 변환하지 마세요.

**날짜 규칙:**
- 날짜는 반드시 YYYY-MM-DD 형식으로 변환하세요.
- 오늘 날짜: {today}
- 미래날짜는 허용하지 않습니다. [현재 메시지]에 미래 날짜가 언급되면 `null`로 반환하세요.
- [현재 메시지]에 날짜 언급이 없으면 반드시 `null`로 반환하세요.

반드시 아래 JSON 형식으로만 응답하세요 (다른 텍스트 없이):
{{"company": "...", "item": "...", "amount": "...", "date": "..."}}

기존 입력 데이터: {existing_slots}
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

def build_current_state_from_slots(invoice_slots: InvoiceSlots | None, status: str) -> str:
    """현재 상태를 IntentRouter에 전달할 한 줄 설명으로 변환."""
    if invoice_slots and (invoice_slots.company or invoice_slots.item or invoice_slots.amount) and status in ("extracting", ""):
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
    for i, invoice_slots in enumerate(invoices):
        _or = lambda v: v if v else "(없음)"
        lines.append(
            f"[{i}번째 전] 거래처={_or(invoice_slots.company)}, 품목={_or(invoice_slots.item)}, "
            f"금액={_or(invoice_slots.amount)}, 날짜={_or(invoice_slots.date)}"
        )
    return "\n".join(lines)
