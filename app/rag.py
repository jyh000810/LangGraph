"""
Qdrant 벡터 검색 — 거래처명 유사도 검색

거래처명을 임베딩 → Qdrant에서 유사도 검색 → EXACT_MATCH / MULTIPLE / NO_MATCH 분류
"""

from __future__ import annotations

import logging

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)

from app.models import CompanyCandidate, CompanySearchResult, MatchStatus

logger = logging.getLogger(__name__)

COLLECTION_NAME = "bill_trans_master_bge-m3"
EXACT_MATCH_THRESHOLD = 0.85
MIN_MATCH_THRESHOLD = 0.6
MAX_RESULTS = 3


async def search_company(
    qdrant: AsyncQdrantClient,
    embed_fn,
    company_name: str,
    user_ccode: str,
) -> CompanySearchResult:
    """거래처명으로 벡터 검색 수행."""
    try:
        embedding = await embed_fn(company_name)

        results = await qdrant.search_groups(
            collection_name=COLLECTION_NAME,
            query_vector=("company_vector", embedding),
            query_filter=Filter(
                must=[FieldCondition(key="CCODE", match=MatchValue(value=user_ccode))]
            ),
            group_by="COMPANY",
            limit=MAX_RESULTS,
            group_size=1,
            score_threshold=MIN_MATCH_THRESHOLD,
            with_payload=True,
        )

        if not results.groups:
            logger.info("[RAG] 거래처 '%s' 매칭 없음 (ccode=%s)", company_name, user_ccode)
            return CompanySearchResult(status=MatchStatus.NO_MATCH)

        candidates = []
        for group in results.groups:
            if not group.hits:
                continue

            sorted_hits = sorted(
                group.hits,
                key=lambda x: (x.score, x.payload.get("BILLSEQ", "")),
                reverse=True,
            )
            pt = sorted_hits[0]

            candidates.append(
                CompanyCandidate(
                    seq=pt.payload.get("BILLSEQ", ""),
                    type="MA",
                    company=pt.payload.get("COMPANY", ""),
                    score=pt.score,
                )
            )

        if not candidates:
            return CompanySearchResult(status=MatchStatus.NO_MATCH)

        if candidates[0].score >= EXACT_MATCH_THRESHOLD:
            logger.info(
                "[RAG] 거래처 '%s' 정확 매칭: %s (score=%.3f)",
                company_name, candidates[0].company, candidates[0].score,
            )
            return CompanySearchResult(status=MatchStatus.EXACT_MATCH, candidates=[candidates[0]])

        logger.info("[RAG] 거래처 '%s' 다중 후보: %d건", company_name, len(candidates))
        return CompanySearchResult(status=MatchStatus.MULTIPLE_MATCHES, candidates=candidates)

    except Exception:
        logger.exception("[RAG] 거래처 검색 실패: %s", company_name)
        return CompanySearchResult(status=MatchStatus.NO_MATCH)
