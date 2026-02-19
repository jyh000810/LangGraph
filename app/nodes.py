"""
LangGraph 노드 함수 — interrupt()를 활용한 human-in-the-loop

노드 시그니처: (state: InvoiceState, config: RunnableConfig) -> dict
interrupt() 노드: 사용자 입력이 필요할 때 그래프를 일시 중단한다.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone, timedelta

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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
    llm = config["configurable"]["llm"]

    # 히스토리 및 현재 슬롯 상태 구성
    messages = state.get("messages") or []
    history_text = build_history_text_from_messages(messages)
    slots = state.get("slots")
    status = state.get("status", "")
    current_state = build_current_state_from_slots(slots, status)

    prompt = INTENT_ROUTER_SYSTEM.format(
        current_state=current_state,
        chat_history=history_text,
    )

    try:
        # LLM 호출 (의도 분류)
        resp = await llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=question),
        ])
        raw = str(resp.content).strip().upper()
        raw_clean = re.sub(r"[^A-Z_]", "", raw)
        logger.info("[IntentAgent] LLM 응답 분석: %s", raw_clean)

        # 정확한 매칭 시도
        for intent in UserIntent:
            if intent.value == raw_clean:
                return {
                    "messages": [HumanMessage(content=question)],
                    "intent": intent.value,
                }

        # 부분 매칭 시도
        for intent in UserIntent:
            if intent.value in raw:
                return {
                    "messages": [HumanMessage(content=question)],
                    "intent": intent.value,
                }

        # 분류 실패 시 기본값 (OTHER)
        logger.warning("[IntentAgent] 의도 분류 모호함 -> OTHER 기본값 할당")
        return {
            "messages": [HumanMessage(content=question)],
            "intent": UserIntent.OTHER.value,
        }

    except Exception as e:
        logger.error("[IntentAgent] 의도 분석 중 오류 발생: %s", e)
        # 예외를 던지면 main.py의 핸들러에서 처리됨 (Step 1)
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
    existing = state.get("slots") or InvoiceSlots()
    
    # 프롬프트 구성 (현재 날짜, 이전 슬롯, 대화 이력 포함)
    existing_str = existing.model_dump_json()
    messages = state.get("messages") or []
    history_text = build_history_text_from_messages(messages)
    today = _today()

    prompt = SLOT_EXTRACTOR_SYSTEM.format(
        today=today,
        existing_slots=existing_str,
        chat_history=history_text,
    )

    try:
        # LLM 호출 (슬롯 추출)
        resp = await llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=question),
        ])
        raw = str(resp.content).strip()
        logger.info("[ExtractionAgent] 추출 시도 LLM 응답: %s", raw)

        # JSON 응답 파싱
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            logger.warning("[ExtractionAgent] JSON 형식의 응답을 찾을 수 없음 -> 기존 상태 유지")
            return {"slots": existing, "status": "extracting"}

        data = json.loads(match.group())
        # 불필요한 null 문자열 제거 및 정규화
        cleaned = {k: (v if v and str(v).lower() != "null" else None) for k, v in data.items()}

        # 금액 정보가 포함된 경우 한글/숫자 혼합 포맷 파싱 (Step 1: 유효성 보강)
        if cleaned.get("amount"):
            parsed_amount = parse_korean_amount(str(cleaned["amount"]))
            if parsed_amount:
                cleaned["amount"] = parsed_amount

        # 신규 추출 데이터와 기존 데이터 병합
        extracted = InvoiceSlots(**cleaned)
        merged = existing.merge(extracted)
        
        logger.info("[ExtractionAgent] 슬롯 병합 성공: 거래처=%s, 금액=%s", merged.company, merged.amount)
        return {"slots": merged, "status": "extracting"}

    except json.JSONDecodeError as e:
        logger.error("[ExtractionAgent] JSON 파싱 실패: %s", e)
        raise LLMResponseError("추출된 데이터 형식이 올바르지 않습니다.") from e
    except Exception as e:
        logger.exception("[ExtractionAgent] 알 수 없는 오류 발생: %s", e)
        # 중요하지 않은 오류인 경우 기존 슬롯 반환으로 대체 가능하나, 여기서는 에러 발생 시킴
        raise AgentError("정보 추출 중 시스템 오류가 발생했습니다.") from e


# ═══════════════════════════════════════════════════════════
#  3. search_company — Qdrant RAG 검색
# ═══════════════════════════════════════════════════════════

async def search_company(state: InvoiceState, config: RunnableConfig) -> dict:
    """거래처명으로 RAG 검색을 수행한다. seq가 이미 확정된 경우 건너뛴다."""
    slots = state.get("slots")

    # seq 확정된 경우 → 이미 거래처 확인됨
    if slots and slots.seq:
        logger.info("[search_company] 거래처 이미 확정 → 건너뜀 (seq=%s)", slots.seq)
        return {"status": "confirmed"}

    # 거래처명 없음 → check_slots에서 처리
    if not slots or not slots.company:
        logger.info("[search_company] 거래처명 없음 → check_slots로")
        return {"status": "no_company"}

    qdrant = config["configurable"].get("qdrant")
    embed_fn = config["configurable"].get("embed_fn")
    user_id = state.get("user_id", "")

    if not qdrant or not embed_fn:
        logger.warning("[search_company] Qdrant/embed_fn 없음 → 바로 check_slots")
        return {"status": "confirmed"}

    result = await rag.search_company(qdrant, embed_fn, slots.company, user_id)

    if result.status == MatchStatus.EXACT_MATCH:
        best = result.best_match()
        updated = slots.model_copy(update={"company": best.company, "seq": best.seq})
        logger.info("[search_company] 정확 매칭: %s (%s)", best.company, best.seq)
        return {"slots": updated, "status": "exact_match"}

    elif result.status == MatchStatus.MULTIPLE_MATCHES:
        logger.info("[search_company] 다중 후보: %d건", len(result.candidates))
        return {"candidates": result.candidates, "status": "multiple_matches"}

    else:
        logger.info("[search_company] 매칭 없음: %s", slots.company)
        return {"status": "no_match"}


def decide_after_search(state: InvoiceState) -> str:
    """search_company 결과에 따라 다음 노드를 결정한다."""
    status = state.get("status", "")
    if status in ("exact_match", "confirmed", "no_company"):
        return "check_slots"
    if status == "multiple_matches":
        return "company_choice"
    return "ask_company"   # no_match


# ═══════════════════════════════════════════════════════════
#  4. company_choice — interrupt(): 다중 후보 선택
# ═══════════════════════════════════════════════════════════

def company_choice(state: InvoiceState) -> dict:
    """다중 거래처 후보를 제시하고 사용자 선택을 기다린다. (interrupt)"""
    candidates = state.get("candidates") or []

    lines = ["유사한 거래처가 여러 건 검색되었습니다. 번호 또는 거래처명으로 선택해주세요."]
    for i, c in enumerate(candidates, 1):
        lines.append(f"{i}. {c.company}")
    lines.append("원하는 거래처가 없으면 정확한 거래처명을 입력해주세요.")
    prompt_msg = "\n".join(lines)

    # ── interrupt: 사용자 입력 대기 ──
    user_answer = interrupt({
        "type": "need_client_choice",
        "message": prompt_msg,
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
    if matched:
        slots = state.get("slots") or InvoiceSlots()
        updated = slots.model_copy(update={"company": matched.company, "seq": matched.seq})
        logger.info("[company_choice] 선택: %s (%s)", matched.company, matched.seq)
        return {
            "messages": [HumanMessage(content=user_answer)],
            "slots": updated,
            "candidates": [],
            "status": "matched",
        }

    # 매칭 실패 → 새 거래처명으로 처리 (extract_slots로 루프)
    logger.info("[company_choice] 매칭 실패 → 새 거래처명: %s", user_answer)
    return {
        "messages": [HumanMessage(content=user_answer)],
        "question": user_answer,
        "candidates": [],
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
#  5. ask_company — interrupt(): 거래처 없음 → 재입력 요청
# ═══════════════════════════════════════════════════════════

def ask_company(state: InvoiceState) -> dict:
    """거래처를 찾을 수 없을 때 정확한 거래처명을 요청한다. (interrupt)"""
    slots = state.get("slots")
    company = slots.company if slots else None
    if company:
        msg = f"'{company}' 거래처를 찾을 수 없습니다. 정확한 거래처명을 입력해주세요."
    else:
        msg = "거래처명을 입력해주세요."

    # ── interrupt: 사용자 입력 대기 ──
    user_answer = interrupt({
        "type": "ask_company",
        "message": msg,
    })

    # 취소 확인
    if _is_cancel(user_answer):
        return {
            "messages": [HumanMessage(content=user_answer)],
            "status": "cancelled",
        }

    logger.info("[ask_company] 사용자 입력: %s", user_answer)

    # seq를 초기화하여 다시 RAG 검색하도록 유도
    updated_slots = slots.model_copy(update={"company": None, "seq": None}) if slots else InvoiceSlots()

    return {
        "messages": [HumanMessage(content=user_answer)],
        "question": user_answer,
        "slots": updated_slots,
        "status": "extracting",
    }


def decide_after_ask_company(state: InvoiceState) -> str:
    """ask_company 후 항상 extract_slots로 루프하거나 취소한다."""
    if state.get("status") == "cancelled":
        return "cancel"
    return "extract_slots"


# ═══════════════════════════════════════════════════════════
#  6. check_slots — 슬롯 완성도 검사 + interrupt(): 부족 시 재입력
# ═══════════════════════════════════════════════════════════

def check_slots(state: InvoiceState) -> dict:
    """슬롯 완성도를 확인하고, 부족하면 사용자 입력을 요청한다. (interrupt)"""
    slots = state.get("slots")

    if slots and slots.is_complete():
        logger.info("[check_slots] 슬롯 완성")
        return {"status": "complete"}

    missing = slots.missing_fields() if slots else ["거래처", "품목", "금액"]
    slots_dict = slots.model_dump() if slots else {}
    msg = f"{', '.join(missing)}을(를) 입력해주세요."

    # ── interrupt: 사용자 입력 대기 ──
    user_answer = interrupt({
        "type": "need_more_info",
        "message": msg,
        "missing": missing,
        "slots": slots_dict,
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
    slots = state.get("slots") or InvoiceSlots()
    redis = config["configurable"].get("redis")
    user_id = state.get("user_id", "")

    if not slots.date:
        slots = slots.model_copy(update={"date": _today()})

    if redis and user_id:
        await save_invoice_to_redis(redis, user_id, slots)

    msg = f"세금계산서 데이터가 준비되었습니다.\n{slots.summary()}"
    logger.info("[finalize] 완료 — %s", slots.summary())

    return {
        "messages": [AIMessage(content=msg)],
        "slots": None,
        "candidates": [],
        "status": "complete",
        "response_type": "data",
        "response_message": msg,
        "response_action": "ready_invoice",
        "response_data": {"slots": slots.model_dump()},
    }


# ═══════════════════════════════════════════════════════════
#  8. select_history — 과거 발행 이력 재사용
# ═══════════════════════════════════════════════════════════

async def select_history(state: InvoiceState, config: RunnableConfig) -> dict:
    """과거 발행 이력에서 적절한 항목을 선택하여 반환한다."""
    redis = config["configurable"].get("redis")
    llm = config["configurable"]["llm"]
    user_id = state.get("user_id", "")
    question = state.get("question", "")

    if not redis:
        msg = "이력 조회 기능을 사용할 수 없습니다. 거래처/품목/금액을 알려주세요."
        return {
            "messages": [AIMessage(content=msg)],
            "slots": None,
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
            "slots": None,
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

    return {
        "messages": [AIMessage(content=msg)],
        "slots": None,
        "candidates": [],
        "status": "complete",
        "response_type": "data",
        "response_message": msg,
        "response_action": "ready_invoice",
        "response_data": {"slots": selected.model_dump()},
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
        "slots": None,
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
    slots = state.get("slots")
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
    if slots and (slots.company or slots.item or slots.amount):
        filled = []
        for key, label in [("company", "거래처"), ("item", "품목"), ("amount", "금액"), ("date", "날짜")]:
            val = getattr(slots, key, None)
            if val:
                filled.append(f"{label}: {val}")
        missing = slots.missing_fields()

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
    """사용자 입력을 후보 목록과 매칭한다.

    우선순위:
    1. 완전 일치 (공백·대소문자 무시)
    2. 사용자 입력이 거래처명에 포함
    3. 거래처명이 사용자 입력에 포함
    """
    user_stripped = user_input.strip()

    # 1순위: 완전 일치
    for c in candidates:
        if user_clean == c.company.replace(" ", "").lower():
            logger.info("[매칭-완전일치] '%s' → %s", user_input, c.company)
            return c

    logger.info("[매칭-실패] '%s'", user_input)
    return None
