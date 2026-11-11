"""差异化修复模块
按成因分类执行不同修复动作，防止错配操作污染证据：
  - RETRIEVAL_MISS   → 查询改写（换角度重检索）
  - EVIDENCE_INSUFFICIENT → 扩量召回（同查询更多）
  - VERSION_MISMATCH → 版本过滤（加元数据约束重检索）
"""
from __future__ import annotations
from typing import Optional

from loguru import logger

from ..models import Chunk, HallucinationCause, SentenceVerification
from ..retrieval.retriever import HybridRetriever


class RepairAction:
    """修复动作的执行结果"""
    def __init__(self, cause: HallucinationCause, new_chunks: list[Chunk], rewritten_query: str = ""):
        self.cause = cause
        self.new_chunks = new_chunks
        self.rewritten_query = rewritten_query
        self.success = len(new_chunks) > 0


class RepairDispatcher:
    """调度层：按成因分发修复动作"""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def repair(
        self,
        original_query: str,
        failed_sv: SentenceVerification,
        current_chunks: list[Chunk],
        iteration: int,
    ) -> RepairAction:
        cause = failed_sv.cause

        if cause == HallucinationCause.RETRIEVAL_MISS:
            return self._repair_retrieval_miss(original_query, failed_sv.sentence)

        elif cause == HallucinationCause.EVIDENCE_INSUFFICIENT:
            return self._repair_evidence_insufficient(original_query, failed_sv.sentence)

        elif cause == HallucinationCause.VERSION_MISMATCH:
            return self._repair_version_mismatch(original_query, failed_sv.sentence, current_chunks)

        else:
            # 未知成因，兜底扩量
            return self._repair_evidence_insufficient(original_query, failed_sv.sentence)

    def _repair_retrieval_miss(self, query: str, failed_sentence: str) -> RepairAction:
        """查询改写：将失败句子的核心信息融入查询"""
        rewritten = self.retriever.rewrite_query(query, failed_sentence)
        logger.debug(f"[REPAIR] RETRIEVAL_MISS → rewrite: '{rewritten}'")
        new_chunks = self.retriever.retrieve(rewritten, expand=False)
        return RepairAction(HallucinationCause.RETRIEVAL_MISS, new_chunks, rewritten)

    def _repair_evidence_insufficient(self, query: str, failed_sentence: str) -> RepairAction:
        """扩量召回：同查询但返回更多结果"""
        logger.debug("[REPAIR] EVIDENCE_INSUFFICIENT → expand recall")
        new_chunks = self.retriever.retrieve(query, expand=True)
        return RepairAction(HallucinationCause.EVIDENCE_INSUFFICIENT, new_chunks)

    def _repair_version_mismatch(
        self, query: str, failed_sentence: str, current_chunks: list[Chunk]
    ) -> RepairAction:
        """版本过滤：提取句子中的版本/型号信息加入元数据过滤条件"""
        import re
        version_pat = re.compile(r"[vV]?\d+\.\d+[a-zA-Z]?|\b[A-Z]{1,3}\d{2,6}\b")
        versions = version_pat.findall(failed_sentence)

        if not versions:
            # 无法提取版本，回退到扩量召回
            return self._repair_evidence_insufficient(query, failed_sentence)

        # 取第一个版本作为过滤条件（实际可支持多版本 OR 逻辑）
        target_version = versions[0]
        logger.debug(f"[REPAIR] VERSION_MISMATCH → version filter: {target_version}")

        where_filter = {"version": target_version}
        try:
            new_chunks = self.retriever.retrieve(query, version_filter=where_filter, expand=False)
        except Exception:
            # ChromaDB where 过滤失败时回退
            new_chunks = self.retriever.retrieve(query, expand=True)

        return RepairAction(HallucinationCause.VERSION_MISMATCH, new_chunks)

    def batch_repair(
        self,
        original_query: str,
        failed_svs: list[SentenceVerification],
        current_chunks: list[Chunk],
        iteration: int,
    ) -> dict[HallucinationCause, list[Chunk]]:
        """
        按成因分组批量修复，避免相同成因重复发起请求。
        返回成因 → 新 chunks 的映射。
        """
        cause_groups: dict[HallucinationCause, list[SentenceVerification]] = {}
        for sv in failed_svs:
            if sv.cause:
                cause_groups.setdefault(sv.cause, []).append(sv)

        result: dict[HallucinationCause, list[Chunk]] = {}
        for cause, svs in cause_groups.items():
            # 用该成因中第一个失败句子做代表修复
            action = self.repair(original_query, svs[0], current_chunks, iteration)
            result[cause] = action.new_chunks
            logger.info(f"[REPAIR] iter={iteration} cause={cause.value} → {len(action.new_chunks)} chunks")

        return result
