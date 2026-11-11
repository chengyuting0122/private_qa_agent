"""外层调度主循环
流程：
  初次检索 → 生成初稿 → 细粒度核验
    → 全通过: SUCCESS
    → 有失败: 差异化修复 → 重判受影响句子 → 迭代
      → 超限仍失败: 触发拒答，返回溯源排查报告
"""
from __future__ import annotations
import time
from typing import Optional

from loguru import logger

from ..models import (
    AgentStatus, Chunk, EvidenceLabel, HallucinationCause,
    SentenceVerification, VerificationReport,
)
from ..config import get_config
from ..retrieval.retriever import HybridRetriever, VectorStore
from ..generation.generator import GenerationAgent
from ..verification.verifier import CascadeVerifier
from ..repair.dispatcher import RepairDispatcher


class TrustQAAgent:
    """私域可信问答智能体主入口"""

    def __init__(
        self,
        vector_store: VectorStore,
        use_nli: bool = True,
    ):
        self.cfg = get_config()["agent"]
        self.max_iterations = self.cfg.get("max_iterations", 3)
        self.refuse_on_exhaustion = self.cfg.get("refuse_on_exhaustion", True)
        self.min_supported_ratio = self.cfg.get("min_supported_ratio", 0.7)

        self.retriever = HybridRetriever(vector_store)
        self.generator = GenerationAgent()
        self.verifier = CascadeVerifier(use_nli=use_nli)
        self.repair = RepairDispatcher(self.retriever)

    def query(self, question: str) -> VerificationReport:
        t0 = time.time()
        logger.info(f"[AGENT] Question: {question}")

        # Step 1: 初次检索
        chunks = self.retriever.retrieve(question)
        if not chunks:
            return self._refuse(
                question, "", [], 0,
                "知识库为空或无相关文档，无法作答",
                t0,
            )

        # Step 2: 生成初稿
        try:
            answer = self.generator.generate(question, chunks)
        except Exception as e:
            return self._refuse(question, "", chunks, 0, f"生成服务不可用: {e}", t0)

        logger.debug(f"[GEN] Draft answer ({len(answer)} chars)")

        # Step 3: 迭代核验与修复
        all_chunks = list(chunks)  # 累积证据池
        sentence_states: list[SentenceVerification] = []
        current_answer = answer

        for iteration in range(self.max_iterations + 1):
            # 核验当前回答
            svs = self.verifier.verify_answer(current_answer, all_chunks)
            sentence_states = svs

            failed = [sv for sv in svs if sv.label != EvidenceLabel.SUPPORT]
            supported_ratio = sum(1 for sv in svs if sv.label == EvidenceLabel.SUPPORT) / max(len(svs), 1)

            logger.info(
                f"[VERIFY] iter={iteration} total={len(svs)} "
                f"supported={len(svs)-len(failed)} failed={len(failed)} "
                f"ratio={supported_ratio:.2f}"
            )

            # 判断是否通过
            if not failed or supported_ratio >= self.min_supported_ratio:
                status = AgentStatus.SUCCESS if not failed else AgentStatus.PARTIAL
                return VerificationReport(
                    question=question,
                    original_answer=answer,
                    sentences=svs,
                    final_answer=current_answer,
                    status=status,
                    iterations=iteration,
                    total_time_ms=(time.time() - t0) * 1000,
                )

            # 已达迭代上限
            if iteration >= self.max_iterations:
                break

            # Step 4: 差异化修复
            repair_map = self.repair.batch_repair(question, failed, all_chunks, iteration)
            new_chunks_added = False
            for cause, new_chunks in repair_map.items():
                for c in new_chunks:
                    if c.chunk_id not in {x.chunk_id for x in all_chunks}:
                        all_chunks.append(c)
                        new_chunks_added = True

            if not new_chunks_added:
                logger.warning("[REPAIR] No new chunks found in repair, stopping early")
                break

            # Step 5: 重新生成（只传入失败句子作为修复焦点）
            failed_sentences = [sv.sentence for sv in failed]
            try:
                current_answer = self.generator.refine(
                    question, current_answer, all_chunks, failed_sentences
                )
            except Exception as e:
                logger.error(f"[REFINE] Failed: {e}")
                break

        # 修复耗尽，触发拒答
        if self.refuse_on_exhaustion:
            # 标记所有仍失败的句子
            for sv in sentence_states:
                if sv.label != EvidenceLabel.SUPPORT:
                    sv.repair_iteration = self.max_iterations

            still_failed = [sv for sv in sentence_states if sv.label != EvidenceLabel.SUPPORT]
            cause_summary = {}
            for sv in still_failed:
                k = sv.cause.value if sv.cause else "unknown"
                cause_summary[k] = cause_summary.get(k, 0) + 1

            refuse_reason = (
                f"经 {self.max_iterations} 轮修复，仍有 {len(still_failed)} 句缺乏有效证据。"
                f"成因分布：{cause_summary}。"
                f"建议排查：{'、'.join(cause_summary.keys())}。"
            )
            return self._refuse(question, answer, all_chunks, self.max_iterations, refuse_reason, t0, sentence_states)

        # 不拒答：输出当前最佳（带警告）
        return VerificationReport(
            question=question,
            original_answer=answer,
            sentences=sentence_states,
            final_answer=current_answer + "\n\n⚠️ 部分内容缺乏充分证据，请谨慎参考。",
            status=AgentStatus.PARTIAL,
            iterations=self.max_iterations,
            total_time_ms=(time.time() - t0) * 1000,
        )

    def _refuse(
        self,
        question: str,
        answer: str,
        chunks: list[Chunk],
        iterations: int,
        reason: str,
        t0: float,
        svs: list[SentenceVerification] = None,
    ) -> VerificationReport:
        logger.warning(f"[REFUSE] {reason}")
        return VerificationReport(
            question=question,
            original_answer=answer,
            sentences=svs or [],
            final_answer=None,
            status=AgentStatus.REFUSED,
            iterations=iterations,
            total_time_ms=(time.time() - t0) * 1000,
            refuse_reason=reason,
        )
