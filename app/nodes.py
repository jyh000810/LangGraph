"""
LangGraph 노드 함수 — interrupt()를 활용한 human-in-the-loop

노드 시그니처: (state: InvoiceState, config: RunnableConfig) -> dict
interrupt() 노드: 사용자 입력이 필요할 때 그래프를 일시 중단한다.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone, timedelta

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from app.models import (
    CompanyCandidate,
    InvoiceSlots,
    MatchStatus,
    UserIntent,
)
from app.prompts import (
    INTENT_ROUTER_SYSTEM,
    INVOICE_SELECTOR_SYSTEM,
    OTHER_FALLBACK,
    OTHER_INSTRUCTION,
    SLOT_EXTRACTOR_SYSTEM,
    build_current_state_from_slots,
    build_history_text_from_messages,
    build_invoice_summary,
)
from app.state import InvoiceState
from app import rag
from app.utils import (
    AgentError,
    LLMResponseError,
    load_invoices_from_redis,
    parse_korean_amount,
    save_invoice_to_redis,
)

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))


class _IntentOut(BaseModel):
    intent: UserIntent


class _SlotExtractOut(BaseModel):
    company: str | None = None
    item: str | None = None
    amount: str | None = None
    date: str | None = None

CANCEL_KEYWORDS = {"취소", "그만", "처음부터", "중단", "cancel", "stop"}


def _today() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d")


def _is_cancel(text: str) -> bool:
    t = text.strip().lower()
    return t in CANCEL_KEYWORDS or any(kw in t for kw in CANCEL_KEYWORDS)


# ═══════════════════════════════════════════════════════════
#  [Agent Group: Intent] 사용자의 의도를 분석하고 분류한다.
# ═══════════════════════════════════════════════════════════

async def route_intent(state: InvoiceState, config: RunnableConfig) -> dict:
    """사용자의 입력을 분석하여 ISSUE_INVOICE, SELECT_HISTORY 등으로 분류한다."""
    question = state.get("question", "")
    llm = config["configurable"].get("llm_fast") or config["configurable"]["llm"]

    # 히스토리 및 현재 슬롯 상태 구성
    messages = state.get("messages") or []
    history_text = build_history_text_from_messages(messages)
    invoice_slots = state.get("invoice_slots") or InvoiceSlots()
    status = state.get("status", "")
    current_state = build_current_state_from_slots(invoice_slots, status)

    prompt = INTENT_ROUTER_SYSTEM.format(
        current_state=current_state,
        chat_history=history_text,
    )

    try:
        # LLM 호출 (의도 분류) — tool calling 대신 JSON 모드 사용 (phi3.5 등 tool 미지원 모델 대응)
        structured_llm = llm.with_structured_output(_IntentOut, method="json_mode")
        resp = await structured_llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=question),
        ])
        logger.info("[IntentAgent] 의도 분류: %s", resp.intent.value)
        return {
            "messages": [HumanMessage(content=question)],
            "intent": resp.intent.value,
        }

    except Exception as e:
        logger.error("[IntentAgent] 의도 분석 중 오류 발생: %s", e)
        raise LLMResponseError("사용자 의도를 파악하는 중 문제가 발생했습니다.") from e


def decide_intent(state: InvoiceState) -> str:
    """route_intent 결과에 따라 다음 노드를 결정한다."""
    return state.get("intent", UserIntent.OTHER.value).lower()


# ═══════════════════════════════════════════════════════════
#  [Agent Group: Extraction] 대화 내용에서 슬롯 정보를 추출한다.
# ═══════════════════════════════════════════════════════════

async def extract_slots(state: InvoiceState, config: RunnableConfig) -> dict:
    """사용자의 발화에서 거래처, 품목, 금액, 날짜 정보를 추출하여 기존 정보와 병합한다."""
    question = state.get("question", "")
    llm = config["configurable"]["llm"]
    existing = state.get("invoice_slots") or InvoiceSlots()
    
    # 프롬프트 구성 (현재 날짜, 이전 슬롯만 전달 — 이력 오염 방지)
    existing_str = existing.model_dump_json()
    today = _today()

    prompt = SLOT_EXTRACTOR_SYSTEM.format(
        today=today,
        existing_slots=existing_str,
    )

    try:
        # LLM 호출 (슬롯 추출) — tool calling 대신 JSON 모드 사용 (스키마 강제)
        structured_llm = llm.with_structured_output(_SlotExtractOut, method="json_mode")
        resp = await structured_llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=question),
        ])
        
        logger.info("[ExtractionAgent] 추출 결과: %s", resp)

        # 금액 한글/숫자 혼합 포맷 파싱
        amount = resp.amount
        if amount:
            parsed_amount = parse_korean_amount(str(amount))
            if parsed_amount:
                amount = parsed_amount

        # 신규 추출 데이터와 기존 데이터 병합
        extracted = InvoiceSlots(company=resp.company, item=resp.item, amount=amount, date=resp.date)
        merged = existing.merge(extracted)

        logger.info("[ExtractionAgent] 슬롯 병합 성공: 거래처=%s, 금액=%s", merged.company, merged.amount)
        return {"invoice_slots": merged, "status": "extracting"}

    except Exception as e:
        logger.exception("[ExtractionAgent] 알 수 없는 오류 발생: %s", e)
        raise AgentError("정보 추출 중 시스템 오류가 발생했습니다.") from e


# ═══════════════════════════════════════════════════════════
#  3. search_company — Qdrant RAG 검색
# ═══════════════════════════════════════════════════════════

async def search_company(state: InvoiceState, config: RunnableConfig) -> dict:
    """거래처명으로 RAG 검색을 수행한다."""
    invoice_slots = state.get("invoice_slots")

    # 거래처명 없음 → check_slots에서 처리
    if not invoice_slots or not invoice_slots.company:
        logger.info("[search_company] 거래처명 없음 → check_slots로")
        return {"status": "no_company"}

    qdrant = config["configurable"].get("qdrant")
    embed_fn = config["configurable"].get("embed_fn")
    user_id = state.get("user_id", "")

    if not qdrant or not embed_fn:
        logger.warning("[search_company] Qdrant/embed_fn 없음 → 바로 check_slots")
        return {"status": "confirmed"}

    result = await rag.search_company(qdrant, embed_fn, invoice_slots.company, user_id)

    if result.status == MatchStatus.EXACT_MATCH:
        best = result.best_match()
        updated = invoice_slots.model_copy(update={"company": best.company, "seq": best.seq})
        logger.info("[search_company] 정확 매칭: %s (%s)", best.company, best.seq)
        return {"invoice_slots": updated, "company_candidate": best, "status": "exact_match"}

    elif result.status == MatchStatus.MULTIPLE_MATCHES:
        logger.info("[search_company] 다중 후보: %d건", len(result.candidates))
        return {"candidates": result.candidates, "status": "multiple_matches"}

    else:
        logger.info("[search_company] 매칭 없음: %s", invoice_slots.company)
        return {"status": "no_match"}


def decide_after_search(state: InvoiceState) -> str:
    """search_company 결과에 따라 다음 노드를 결정한다."""
    status = state.get("status", "")
    if status in ("exact_match", "confirmed", "no_company"):
        return "check_slots"
    else:
        return "company_choice"


# ═══════════════════════════════════════════════════════════
#  4. company_choice — interrupt(): 다중 후보 선택
# ═══════════════════════════════════════════════════════════

def company_choice(state: InvoiceState) -> dict:
    """다중 거래처 후보를 제시하고 사용자 선택을 기다린다. (interrupt)"""
    candidates = state.get("candidates") or []

    prompt_msg = "유사한 거래처가 여러 건 검색되었습니다. 거래처명으로 선택해주세요. 원하는 거래처가 없으면 정확한 거래처명을 입력해주세요."
    msg = "거래처명을 입력해주세요."

    # ── interrupt: 사용자 입력 대기 ──
    user_answer = interrupt({
        "type": "need_client_choice",
        "message": prompt_msg if candidates else msg,
        "candidates": [c.model_dump() for c in candidates],
    })

    # 취소 확인
    if _is_cancel(user_answer):
        return {
            "messages": [HumanMessage(content=user_answer)],
            "candidates": [],
            "status": "cancelled",
        }

    # 후보 매칭 시도
    matched = _match_company_from_candidates(user_answer, candidates)
    invoice_slots = state.get("invoice_slots") or InvoiceSlots()
    if matched:
        updated = invoice_slots.model_copy(update={"company": matched.company, "seq": matched.seq})
        logger.info("[company_choice] 선택: %s (%s)", matched.company, matched.seq)
        return {
            "messages": [HumanMessage(content=user_answer)],
            "invoice_slots": updated,
            "company_candidate": matched,
            "candidates": [],
            "status": "matched",
        }

    # 매칭 실패 → 새 거래처명으로 처리 (extract_slots로 루프)
    logger.info("[company_choice] 매칭 실패 → 새 거래처명: %s", user_answer)
    # seq를 초기화하여 다시 RAG 검색하도록 유도
    updated_slots = invoice_slots.model_copy(update={"company": user_answer}) if invoice_slots else InvoiceSlots()
    return {
        "messages": [HumanMessage(content=user_answer)],
        "question": user_answer,
        "invoice_slots": updated_slots,
        "status": "extracting",
    }


def decide_after_company_choice(state: InvoiceState) -> str:
    """company_choice 결과에 따라 다음 노드를 결정한다."""
    status = state.get("status", "")
    if status == "matched":
        return "check_slots"
    if status == "cancelled":
        return "cancel"
    return "extract_slots"   # extracting → 루프


# ═══════════════════════════════════════════════════════════
#  6. search_item_amount — 품목/금액 RAG 검색 (통합)
#
#  케이스별 동작:
#    품목 O, 금액 O  → confirmed (통과)
#    품목 X, 금액 O  → 금액 일치 이력 최대 3건 선택지 제공
#    품목 O, 금액 X  → 품목 유사도 0.7 이상 이력 선택지 제공
#    품목 X, 금액 X  → 최근 거래 기반 이력 선택지 제공
#
#  RAG는 사용자가 해당 필드를 입력하지 않은 경우에만 실행된다.
# ═══════════════════════════════════════════════════════════

async def search_item_amount(state: InvoiceState, config: RunnableConfig) -> dict:
    """품목/금액 조합에 따라 적절한 RAG 검색을 수행하고 후보를 반환한다."""
    invoice_slots = state.get("invoice_slots")
    has_item = bool(invoice_slots and invoice_slots.item)
    has_amount = bool(invoice_slots and invoice_slots.amount)

    # 둘 다 입력됨 → 통과
    if has_item and has_amount:
        logger.info("[search_item_amount] 품목/금액 모두 확인 → 통과")
        return {"status": "confirmed"}

    # qdrant = config["configurable"].get("qdrant")
    # embed_fn = config["configurable"].get("embed_fn")
    # user_id = state.get("user_id", "")
    # redis = config["configurable"].get("redis")

    if not has_item and has_amount:
        # 금액 일치 이력 최대 3건 검색
        logger.info("[search_item_amount] 품목 없음 → 금액(%s) 기반 검색", invoice_slots.amount)
        # candidates = await rag.search_by_amount(qdrant, embed_fn, slots.amount, user_id, limit=3)
        # return {"candidates": candidates, "status": "no_item"}
        return {"candidates": [], "status": "no_item"}  # RAG 구현 전 임시

    if has_item and not has_amount:
        # 품목 유사도 0.7 이상 이력 검색
        logger.info("[search_item_amount] 금액 없음 → 품목(%s) 유사도 기반 검색", invoice_slots.item)
        # candidates = await rag.search_by_item_similarity(qdrant, embed_fn, slots.item, user_id, threshold=0.7)
        # return {"candidates": candidates, "status": "no_amount"}
        return {"candidates": [], "status": "no_amount"}  # RAG 구현 전 임시

    # 둘 다 없음 → 최근 거래 이력 제공
    logger.info("[search_item_amount] 품목/금액 모두 없음 → 최근 거래 기반 검색")
    # past_invoices = await load_invoices_from_redis(redis, user_id, limit=3)
    # candidates = [{"item": s.item, "amount": s.amount} for s in past_invoices if s.item or s.amount]
    # return {"candidates": candidates, "status": "no_item_no_amount"}
    return {"candidates": [], "status": "no_item_no_amount"}  # RAG 구현 전 임시


def decide_after_search_item_amount(state: InvoiceState) -> str:
    """search_item_amount 결과에 따라 다음 노드를 결정한다."""
    if state.get("status") == "confirmed":
        return "check_slots"
    return "item_amount_choice"


def item_amount_choice(state: InvoiceState) -> dict:
    """품목/금액 후보를 제시하고 사용자 선택 또는 직접 입력을 처리한다. (interrupt)

    - 후보 선택 시: 해당 이력의 품목/금액을 슬롯에 반영
    - 매칭 실패 시: 입력값을 question으로 전달하여 extract_slots로 루프
    """
    candidates = state.get("candidates") or []
    status = state.get("status", "")
    invoice_slots = state.get("invoice_slots") or InvoiceSlots()

    # 케이스별 안내 메시지
    if status == "no_item":
        prompt_msg = f"금액 {invoice_slots.amount}에 해당하는 이전 거래 내역입니다. 품목을 선택하거나 직접 입력해주세요."
    elif status == "no_amount":
        prompt_msg = f"품목 '{invoice_slots.item}'과 유사한 이전 거래 내역입니다. 금액을 선택하거나 직접 입력해주세요."
    else:  # no_item_no_amount
        prompt_msg = "최근 거래 내역입니다. 재사용하거나 품목/금액을 직접 입력해주세요."

    user_answer = interrupt({
        "type": "need_item_amount_choice",
        "message": prompt_msg,
        "search_status": status,
        "candidates": candidates,
    })

    if _is_cancel(user_answer):
        return {
            "messages": [HumanMessage(content=user_answer)],
            "candidates": [],
            "status": "cancelled",
        }

    update = {}
    if status == "no_item":
        update["item"] = user_answer if user_answer else invoice_slots.item
    elif status == "no_amount":
        update["amount"] = user_answer if user_answer else invoice_slots.amount
    else:  # no_item_no_amount
        update["item"] = user_answer if user_answer else invoice_slots.item
        update["amount"] = user_answer if user_answer else invoice_slots.amount
        
    updated_slots = invoice_slots.model_copy(update=update)
    
    return {
        "messages": [HumanMessage(content=user_answer)],
        "invoice_slots": updated_slots,
        "candidates": [],
        "status": "matched",
    }


def decide_after_item_amount_choice(state: InvoiceState) -> str:
    """item_amount_choice 결과에 따라 다음 노드를 결정한다."""
    status = state.get("status", "")
    if status == "matched":
        return "check_slots"
    if status == "cancelled":
        return "cancel"
    return "extract_slots"   # extracting → 루프


# ═══════════════════════════════════════════════════════════
#  7. check_slots — 슬롯 완성도 검사 + interrupt(): 부족 시 재입력
# ═══════════════════════════════════════════════════════════

def check_slots(state: InvoiceState) -> dict:
    """슬롯 완성도를 확인하고, 부족하면 사용자 입력을 요청한다. (interrupt)"""
    invoice_slots = state.get("invoice_slots")

    if invoice_slots and invoice_slots.is_complete():
        logger.info("[check_slots] 슬롯 완성")
        return {"status": "complete"}

    missing = invoice_slots.missing_fields() if invoice_slots else ["거래처", "품목", "금액"]
    slots_dict = invoice_slots.model_dump() if invoice_slots else {}
    msg = f"{', '.join(missing)}을(를) 입력해주세요."

    # ── interrupt: 사용자 입력 대기 ──
    user_answer = interrupt({
        "type": "need_more_info",
        "message": msg,
        "missing": missing,
        "invoice_slots": slots_dict,
    })

    # 취소 확인
    if _is_cancel(user_answer):
        return {
            "messages": [HumanMessage(content=user_answer)],
            "status": "cancelled",
        }

    logger.info("[check_slots] 추가 입력: %s", user_answer)
    return {
        "messages": [HumanMessage(content=user_answer)],
        "question": user_answer,
        "status": "extracting",
    }


def decide_after_check_slots(state: InvoiceState) -> str:
    """check_slots 결과에 따라 다음 노드를 결정한다."""
    status = state.get("status", "")
    if status == "complete":
        return "finalize"
    if status == "cancelled":
        return "cancel"
    return "extract_slots"   # extracting → 루프


# ═══════════════════════════════════════════════════════════
#  7. finalize — 세금계산서 데이터 준비 완료
# ═══════════════════════════════════════════════════════════

async def finalize(state: InvoiceState, config: RunnableConfig) -> dict:
    """슬롯 완성 → Redis에 이력 저장 → ready_invoice 응답."""
    invoice_slots = state.get("invoice_slots") or InvoiceSlots()
    company_candidate = state.get("company_candidate")

    if not invoice_slots.date:
        invoice_slots = invoice_slots.model_copy(update={"date": _today()})

    msg = f"세금계산서 데이터가 준비되었습니다.\n{invoice_slots.summary()}"
    logger.info("[finalize] 완료 — %s", invoice_slots.summary())

    # 거래처 정보와 청구 데이터 분리
    if company_candidate:
        company_data = {
            "seq": company_candidate.seq,
            "type": company_candidate.type,
            "company": company_candidate.company,
        }
    else:
        logger.warning("[finalize] company_candidate가 None — seq 없이 진행")
        company_data = {
            "seq": "",
            "type": None,
            "company": invoice_slots.company or "",
        }
    
    invoice_data = {
        "company": invoice_slots.company,
        "item": invoice_slots.item,
        "amount": invoice_slots.amount,
        "date": invoice_slots.date,
    }

    return {
        "messages": [AIMessage(content=msg)],
        "invoice_slots": None,
        "company_candidate": None,
        "candidates": [],
        "status": "complete",
        "response_type": "data",
        "response_message": msg,
        "response_action": "ready_invoice",
        "response_data": {"company": company_data, "invoice": invoice_data},
    }


# ═══════════════════════════════════════════════════════════
#  8. select_history — 과거 발행 이력 재사용
# ═══════════════════════════════════════════════════════════

async def select_history(state: InvoiceState, config: RunnableConfig) -> dict:
    """과거 발행 이력에서 적절한 항목을 선택하여 반환한다."""
    redis = config["configurable"].get("redis")
    llm = config["configurable"].get("llm_logical") or config["configurable"]["llm"]
    user_id = state.get("user_id", "")
    question = state.get("question", "")

    if not redis:
        msg = "이력 조회 기능을 사용할 수 없습니다. 거래처/품목/금액을 알려주세요."
        return {
            "messages": [AIMessage(content=msg)],
            "invoice_slots": None,
            "candidates": [],
            "response_type": "message",
            "response_message": msg,
            "response_action": None,
            "response_data": None,
        }

    past_invoices = await load_invoices_from_redis(redis, user_id, limit=10)
    if not past_invoices:
        msg = "과거 발행 내역이 없습니다. 거래처/품목/금액을 알려주세요."
        return {
            "messages": [AIMessage(content=msg)],
            "invoice_slots": None,
            "candidates": [],
            "response_type": "message",
            "response_message": msg,
            "response_action": None,
            "response_data": None,
        }

    summary = build_invoice_summary(past_invoices)
    prompt = INVOICE_SELECTOR_SYSTEM.format(invoice_summary=summary)

    try:
        resp = await llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=question),
        ])
        m = re.search(r"\d+", resp.content.strip())
        index = int(m.group()) if m else 0
    except Exception:
        logger.exception("[select_history] LLM 오류")
        index = 0

    if index < 0 or index >= len(past_invoices):
        index = 0

    selected = past_invoices[index].model_copy(update={"date": _today()})
    msg = f"이전 발행 내역을 불러왔습니다.\n{selected.summary()}"
    logger.info("[select_history] index=%d 선택", index)

    company_data = {"seq": "1111", "company": selected.company, "type": "MA"}
    invoice_data = {"company": selected.company, "item": selected.item, "amount": selected.amount, "date": selected.date}

    return {
        "messages": [AIMessage(content=msg)],
        "invoice_slots": None,
        "candidates": [],
        "status": "complete",
        "response_type": "data",
        "response_message": msg,
        "response_action": "ready_invoice",
        "response_data": {"company": company_data, "invoice": invoice_data},
    }


# ═══════════════════════════════════════════════════════════
#  9. cancel — 작업 취소
# ═══════════════════════════════════════════════════════════

def cancel(state: InvoiceState) -> dict:
    """진행 중인 작업을 취소한다."""
    msg = "진행 중이던 작업이 취소되었습니다. 새로운 요청을 입력해주세요."
    logger.info("[cancel] 취소 처리")
    return {
        "messages": [AIMessage(content=msg)],
        "invoice_slots": None,
        "candidates": [],
        "status": "cancelled",
        "response_type": "message",
        "response_message": msg,
        "response_action": None,
        "response_data": None,
    }


# ═══════════════════════════════════════════════════════════
#  10. general_chat — 일반 대화
# ═══════════════════════════════════════════════════════════

async def general_chat(state: InvoiceState, config: RunnableConfig) -> dict:
    """일반 대화 처리. 슬롯 수집 중이면 현재 상태도 안내한다."""
    llm_creative = config["configurable"].get("llm_creative") or config["configurable"]["llm"]
    question = state.get("question", "")
    invoice_slots = state.get("invoice_slots")
    messages = state.get("messages") or []
    history_text = build_history_text_from_messages(messages)

    prompt = f"{OTHER_INSTRUCTION}\n\n대화 이력:\n{history_text}"

    try:
        resp = await llm_creative.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=question),
        ])
        answer = resp.content.strip() or OTHER_FALLBACK
    except Exception:
        logger.exception("[general_chat] LLM 오류")
        answer = OTHER_FALLBACK

    # 슬롯 수집 중이면 현재 상태 안내 추가
    if invoice_slots and (invoice_slots.company or invoice_slots.item or invoice_slots.amount):
        filled = []
        for key, label in [("company", "거래처"), ("item", "품목"), ("amount", "금액"), ("date", "날짜")]:
            val = getattr(invoice_slots, key, None)
            if val:
                filled.append(f"{label}: {val}")
        missing = invoice_slots.missing_fields()

        status_lines = [answer, ""]
        if filled:
            status_lines.append(f"현재 입력된 정보: {', '.join(filled)}")
        if missing:
            status_lines.append(f"아직 부족한 정보: {', '.join(missing)}")
            status_lines.append("위 정보를 알려주시면 세금계산서 발행 데이터를 완성할 수 있어요.")
        answer = "\n".join(status_lines)

    return {
        "messages": [AIMessage(content=answer)],
        "response_type": "message",
        "response_message": answer,
        "response_action": None,
        "response_data": None,
    }


# ═══════════════════════════════════════════════════════════
#  헬퍼 함수
# ═══════════════════════════════════════════════════════════

def _match_company_from_candidates(
    user_input: str,
    candidates: list[CompanyCandidate],
) -> CompanyCandidate | None:
    """사용자 입력을 거래처 후보 목록과 매칭한다. (완전 일치, 공백·대소문자 무시)"""
    user_clean = user_input.strip().replace(" ", "").lower()
    if not user_clean:
        return None
    for c in candidates:
        if user_clean == c.company.replace(" ", "").lower():
            logger.info("[매칭-완전일치] '%s' → %s", user_input, c.company)
            return c
    logger.info("[매칭-실패] '%s'", user_input)
    return None
