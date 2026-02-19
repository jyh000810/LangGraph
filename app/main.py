"""
FastAPI 진입점 — 세금계산서 발행 데이터 세팅 API (LangGraph MemorySaver + interrupt)

실행:
  uvicorn app.main:app --host 0.0.0.0 --port 8200

핵심 동작:
  - POST /chat/{user_id}: 대화 처리
    • 그래프가 interrupt 상태이면 → Command(resume=question) 으로 재개
    • 아니면 → 새 invoke
  - GET /state/{user_id}: 그래프 상태 직접 조회
  - GET /invoices/{user_id}: Redis 발행 이력 조회
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import redis.asyncio as aioredis
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph import StateGraph
from langgraph.types import Command
from qdrant_client import AsyncQdrantClient

from app.config import settings
from app.graph import build_graph
from app.models import ChatRequest, ChatResponse, ChatResult
from app.utils import append_chat_history, load_chat_history, load_invoices_from_redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── 리소스 (lifespan으로 관리) ──

redis_pool: aioredis.Redis | None = None
qdrant_client: AsyncQdrantClient | None = None
llm: ChatOllama | None = None
llm_creative: ChatOllama | None = None
_embeddings: OllamaEmbeddings | None = None
invoice_graph: StateGraph | None = None


def _create_llm(temperature: float) -> ChatOllama:
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model_name,
        temperature=temperature,
        timeout=30,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_pool, qdrant_client, llm, llm_creative, _embeddings, invoice_graph

    qdrant_client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    llm = _create_llm(temperature=0.01)
    llm_creative = _create_llm(temperature=0.7)

    _embeddings = OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
    )

    # AsyncRedisSaver: 서버 재시작 후에도 그래프 상태 복원 + 1주일 TTL 자동 만료
    WEEK_SECONDS = 7 * 24 * 60 * 60   # 604800초
    checkpointer = AsyncRedisSaver(settings.redis_url, ttl={"default": WEEK_SECONDS})
    await checkpointer.asetup()
    invoice_graph = build_graph().compile(checkpointer=checkpointer)

    logger.info("서비스 시작 — Ollama=%s, model=%s", settings.ollama_base_url, settings.ollama_model_name)
    yield

    if redis_pool:
        await redis_pool.aclose()
    if qdrant_client:
        await qdrant_client.close()
    logger.info("서비스 종료")


async def _embed_fn(text: str) -> list[float]:
    return await _embeddings.aembed_query(text)


def _make_config(user_id: str) -> dict:
    """그래프 실행 config 생성. thread_id = user_id → MemorySaver가 자동 저장/복원."""
    return {
        "configurable": {
            "thread_id": user_id,
            "redis": redis_pool,
            "qdrant": qdrant_client,
            "embed_fn": _embed_fn,
            "llm": llm,
            "llm_creative": llm_creative,
        }
    }


app = FastAPI(title="LangGraph Invoice Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_tracking_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    start = time.monotonic()
    response = await call_next(request)
    elapsed = round((time.monotonic() - start) * 1000, 1)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = str(elapsed)
    logger.info("[%s] %s %s — %dms", request_id, request.method, request.url.path, elapsed)
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content=ChatResponse(
            type="message",
            text="처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
        ).model_dump(),
    )


# ── 핵심 엔드포인트 ──

@app.post("/chat/{user_id}", response_model=ChatResponse)
async def chat(user_id: str, req: ChatRequest):
    """메인 엔드포인트 — resume vs fresh invoke 자동 분기."""
    if not user_id or len(user_id) > 100:
        return ChatResponse(type="message", text="사용자 정보가 필요합니다.")
    if not req.question or not req.question.strip():
        return ChatResponse(type="message", text="메시지 내용을 입력해주세요.")
    if len(req.question) > 200:
        return ChatResponse(type="message", text="메시지가 너무 깁니다. 200자 이내로 입력해주세요.")

    question = req.question.strip()
    config = _make_config(user_id)

    logger.info("[Chat] user_id=%s, question=%s", user_id, question[:50])

    # ── 사용자 메시지 Redis에 저장 ──
    await append_chat_history(redis_pool, user_id, "user", question)

    try:
        # ── 현재 그래프 상태 확인 ──
        current_state = await invoice_graph.aget_state(config)

        if current_state.next:
            # interrupt 상태 → resume
            logger.info("[Chat] interrupt 재개 — next=%s", current_state.next)
            result = await asyncio.wait_for(
                invoice_graph.ainvoke(Command(resume=question), config),
                timeout=120,
            )
        else:
            # 새 요청
            logger.info("[Chat] 새 invoke")
            result = await asyncio.wait_for(
                invoice_graph.ainvoke(
                    {"question": question, "user_id": user_id},
                    config,
                ),
                timeout=120,
            )

    except asyncio.TimeoutError:
        logger.error("[Chat] 타임아웃 (120초) — user_id=%s", user_id)
        timeout_msg = "처리 시간이 초과되었습니다. 잠시 후 다시 시도해주세요."
        await append_chat_history(redis_pool, user_id, "assistant", timeout_msg)
        return ChatResponse(type="message", text=timeout_msg)

    # ── 응답 구성: interrupt 여부 확인 ──
    new_state = await invoice_graph.aget_state(config)

    if new_state.next:
        # 방금 interrupt 발생 → interrupt 데이터를 응답으로
        interrupt_data = _extract_interrupt_data(new_state)
        msg = interrupt_data.get("message", "추가 정보가 필요합니다.")
        action = interrupt_data.get("type")
        raw = {k: v for k, v in interrupt_data.items() if k != "message"}

        logger.info("[Chat] interrupt 응답 — type=%s", action)
        await append_chat_history(redis_pool, user_id, "assistant", msg)
        return ChatResponse(
            type="data",
            text=msg,
            result=ChatResult(action=action, raw=raw) if action else None,
        )

    # ── 그래프 완료 → 최종 응답 반환 ──
    action = result.get("response_action")
    raw_data = result.get("response_data")
    response_msg = result.get("response_message", "")

    await append_chat_history(redis_pool, user_id, "assistant", response_msg)
    return ChatResponse(
        type=result.get("response_type", "message"),
        text=response_msg,
        result=ChatResult(action=action, raw=raw_data) if (action or raw_data) else None,
    )


def _extract_interrupt_data(state) -> dict[str, Any]:
    """StateSnapshot에서 interrupt 데이터를 추출한다."""
    try:
        tasks = state.tasks
        if tasks and tasks[0].interrupts:
            return dict(tasks[0].interrupts[0].value)
    except Exception:
        logger.warning("[interrupt] 데이터 추출 실패")
    return {"message": "추가 정보를 입력해주세요."}


# ── 상태 조회 엔드포인트 ──

@app.get("/state/{user_id}")
async def get_graph_state(user_id: str):
    """LangGraph 그래프 상태를 직접 조회한다."""
    if not user_id or len(user_id) > 100:
        return JSONResponse(status_code=400, content={"error": "잘못된 사용자 정보입니다."})

    config = _make_config(user_id)
    state = await invoice_graph.aget_state(config)

    slots = state.values.get("slots")
    return {
        "user_id": user_id,
        "next": list(state.next),
        "interrupted": bool(state.next),
        "slots": slots.model_dump() if slots else None,
        "candidates": [c.model_dump() for c in (state.values.get("candidates") or [])],
        "intent": state.values.get("intent"),
        "status": state.values.get("status"),
        "message_count": len(state.values.get("messages") or []),
    }


# ── 발행 이력 엔드포인트 ──

@app.get("/invoices/{user_id}")
async def get_invoices(user_id: str, limit: int = 10):
    """Redis 발행 이력을 조회한다."""
    if not user_id or len(user_id) > 100:
        return JSONResponse(status_code=400, content={"error": "잘못된 사용자 정보입니다."})
    limit = max(1, min(limit, 50))

    invoices = await load_invoices_from_redis(redis_pool, user_id, limit=limit)
    return {
        "user_id": user_id,
        "count": len(invoices),
        "invoices": [
            {
                "company": inv.company,
                "item": inv.item,
                "amount": inv.amount,
                "date": inv.date,
            }
            for inv in invoices
        ],
    }


# ── 채팅 이력 조회 ──

@app.get("/chat/{user_id}/history")
async def get_chat_history(user_id: str, limit: int = 30):
    """Redis에서 대화 이력을 조회한다."""
    if not user_id or len(user_id) > 100:
        return JSONResponse(status_code=400, content={"error": "잘못된 사용자 정보입니다."})
    limit = max(1, min(limit, 200))

    rows = await load_chat_history(redis_pool, user_id, limit=limit)
    return {
        "user_id": user_id,
        "count": len(rows),
        "messages": [{"role": r.role, "content": r.content} for r in rows],
    }


# ── 세션 초기화 ──

@app.delete("/chat/{user_id}")
async def reset_session(user_id: str):
    """Redis 채팅 이력을 삭제한다. (그래프 상태는 MemorySaver 안에 유지)"""
    if not user_id or len(user_id) > 100:
        return JSONResponse(status_code=400, content={"error": "잘못된 사용자 정보입니다."})

    from app.utils import CHAT_HISTORY_KEY
    key = CHAT_HISTORY_KEY.format(user_id=user_id)
    await redis_pool.delete(key)
    return {"status": "ok", "message": "채팅 이력이 초기화되었습니다.", "user_id": user_id}


# ── 헬스 체크 ──

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/health/detail")
async def health_detail():
    """Redis, Qdrant, Ollama 상세 헬스 체크."""
    checks: dict[str, str] = {}

    try:
        await redis_pool.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"

    try:
        await qdrant_client.get_collections()
        checks["qdrant"] = "ok"
    except Exception as e:
        checks["qdrant"] = f"error: {e}"

    try:
        import httpx
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            checks["ollama"] = "ok" if resp.status_code == 200 else f"status: {resp.status_code}"
    except Exception as e:
        checks["ollama"] = f"error: {e}"

    all_ok = all(v == "ok" for v in checks.values())
    return {"status": "ok" if all_ok else "degraded", "checks": checks}
