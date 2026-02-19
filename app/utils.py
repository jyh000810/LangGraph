"""
유틸리티 함수 — 한글 금액 파싱 + Redis 채팅 이력 + Redis 발행 이력 헬퍼
"""

from __future__ import annotations

import json
import logging
import re
from decimal import Decimal, InvalidOperation
from datetime import datetime

logger = logging.getLogger(__name__)

# ── 커스텀 예외 클래스 (Step 1: 에러 핸들링 강화) ──

class AgentError(Exception):
    """에이전트 실행 중 발생하는 기본 예외"""
    def __init__(self, message: str, detail: str = None):
        super().__init__(message)
        self.message = message
        self.detail = detail

class LLMResponseError(AgentError):
    """LLM 응답 파싱 실패 또는 부적절한 응답 시 발생"""
    pass

class ExternalServiceError(AgentError):
    """Redis, Qdrant 등 외부 서비스 연동 실패 시 발생"""
    pass

# Redis 키 패턴 (기존 프로젝트와 동일)
CHAT_HISTORY_KEY  = "chat:history:{user_id}"
INVOICE_HISTORY_KEY = "chat:invoice:{user_id}"

CHAT_HISTORY_KEEP = 200
CHAT_TTL          = 60 * 60 * 24 * 7   # 7일

INVOICE_KEEP = 20
INVOICE_TTL  = 60 * 60 * 24 * 30       # 30일


# ── 한글 금액 파싱 ──

def parse_korean_amount(text: str) -> str | None:
    """한글/숫자 혼합 금액 표현을 순수 숫자 문자열로 변환한다.

    예시:
        "오천만원"    → "50000000"
        "500만원"     → "5000000"
        "1억5천만"    → "150000000"
        "3만원"       → "30000"
        "삼백만원"    → "3000000"
        "1234"        → "1234"
        ""            → None
    """
    if not text:
        return None

    clean_text = re.sub(r'[^0-9.영일이삼사오육칠팔구십백천만억조]', '', text)

    if not clean_text:
        return None

    if re.match(r'^[0-9.]+$', clean_text):
        try:
            return str(int(Decimal(clean_text)))
        except InvalidOperation:
            return None

    units = {'십': 10, '백': 100, '천': 1000}
    big_units = {'만': 10000, '억': 100000000, '조': 1000000000000}
    nums = {'영': 0, '일': 1, '이': 2, '삼': 3, '사': 4,
            '오': 5, '육': 6, '칠': 7, '팔': 8, '구': 9}

    total = Decimal(0)
    current_group = Decimal(0)
    last_num = Decimal(0)
    num_buffer: list[str] = []

    for char in clean_text:
        if char.isdigit() or char == '.':
            num_buffer.append(char)
        else:
            if num_buffer:
                try:
                    last_num = Decimal("".join(num_buffer))
                except InvalidOperation:
                    last_num = Decimal(0)
                num_buffer = []

            if char in nums:
                last_num = Decimal(nums[char])

            elif char in units:
                unit_val = Decimal(units[char])
                if last_num == 0:
                    last_num = Decimal(1)
                current_group += last_num * unit_val
                last_num = Decimal(0)

            elif char in big_units:
                big_unit_val = Decimal(big_units[char])
                current_group += last_num
                if current_group == 0:
                    current_group = Decimal(1)
                total += current_group * big_unit_val
                current_group = Decimal(0)
                last_num = Decimal(0)

    if num_buffer:
        try:
            last_num = Decimal("".join(num_buffer))
        except InvalidOperation:
            pass

    final_amount = total + current_group + last_num

    if final_amount <= 0:
        return None

    return str(int(final_amount))


# ── Redis 채팅 이력 헬퍼 ──

async def append_chat_history(redis, user_id: str, role: str, content: str) -> None:
    """대화 메시지를 Redis에 추가한다. (최대 200건, TTL 7일)"""
    from app.models import MessageRow

    key = CHAT_HISTORY_KEY.format(user_id=user_id)
    try:
        record = json.dumps(MessageRow(role=role, content=content).model_dump(), ensure_ascii=False)
        pipe = redis.pipeline()
        pipe.rpush(key, record)
        pipe.ltrim(key, -CHAT_HISTORY_KEEP, -1)
        pipe.expire(key, CHAT_TTL)
        await pipe.execute()
    except Exception:
        logger.exception("[ChatHistory] 저장 실패 — user_id=%s", user_id)


async def load_chat_history(redis, user_id: str, limit: int = 30):
    """Redis에서 최근 대화 이력을 로드한다."""
    from app.models import MessageRow

    key = CHAT_HISTORY_KEY.format(user_id=user_id)
    try:
        raw_list = await redis.lrange(key, -limit, -1)
        rows = []
        for raw in raw_list:
            try:
                rows.append(MessageRow.model_validate_json(raw))
            except Exception:
                logger.warning("[ChatHistory] 파싱 오류: %s", raw[:80])
        return rows
    except Exception:
        logger.exception("[ChatHistory] 로드 실패 — user_id=%s", user_id)
        return []


# ── Redis 발행 이력 헬퍼 ──

async def save_invoice_to_redis(redis, user_id: str, slots) -> None:
    """완성된 슬롯을 Redis 발행 이력에 저장한다. (최대 20건, TTL 30일)"""
    key = INVOICE_HISTORY_KEY.format(user_id=user_id)
    try:
        record = json.dumps(slots.model_dump(), ensure_ascii=False)
        pipe = redis.pipeline()
        pipe.rpush(key, record)
        pipe.ltrim(key, -INVOICE_KEEP, -1)
        pipe.expire(key, INVOICE_TTL)
        await pipe.execute()
        logger.info("[InvoiceHistory] 저장 — user_id=%s, company=%s", user_id, slots.company)
    except Exception:
        logger.exception("[InvoiceHistory] 저장 실패 — user_id=%s", user_id)


async def load_invoices_from_redis(redis, user_id: str, limit: int = 10):
    """Redis에서 발행 이력을 로드한다 (최신순)."""
    from app.models import InvoiceSlots

    key = INVOICE_HISTORY_KEY.format(user_id=user_id)
    try:
        raw_list = await redis.lrange(key, -limit, -1)
        result = []
        for raw in raw_list:
            try:
                result.append(InvoiceSlots.model_validate_json(raw))
            except Exception:
                logger.warning("[InvoiceHistory] 파싱 오류: %s", raw[:80])
        result.reverse()   # 최신순
        return result
    except Exception:
        logger.exception("[InvoiceHistory] 로드 실패 — user_id=%s", user_id)
        return []
