"""数据模型定义"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EvidenceLabel(str, Enum):
    """NLI 三分类标签"""
    SUPPORT = "support"       # 证据支持该句
    REFUTE = "refute"         # 证据反驳该句
    INSUFFICIENT = "insufficient"  # 证据不足


class HallucinationCause(str, Enum):
    """幻觉成因分类（用于差异化修复）"""
    RETRIEVAL_MISS = "retrieval_miss"     # 检索未命中：相关文档根本没被召回
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"  # 证据不足：召回了但内容不够
    VERSION_MISMATCH = "version_mismatch"  # 依据不适用：版本/型号不匹配


class AgentStatus(str, Enum):
    """智能体最终状态"""
    SUCCESS = "success"         # 所有句子均有支持证据
    PARTIAL = "partial"         # 部分句子缺失证据但超过阈值
    REFUSED = "refused"         # 修复失败，触发拒答
    ERROR = "error"


@dataclass
class Chunk:
    """文档分块"""
    chunk_id: str
    content: str
    source: str                 # 原始文件路径
    page: Optional[int] = None
    metadata: dict = field(default_factory=dict)  # 型号、版本、章节等
    score: float = 0.0          # 检索得分（用于排序）


@dataclass
class SentenceVerification:
    """单句核验结果"""
    sentence: str
    label: EvidenceLabel
    cause: Optional[HallucinationCause] = None
    supporting_chunks: list[Chunk] = field(default_factory=list)
    rule_check_passed: bool = True
    rule_check_detail: str = ""
    nli_scores: dict = field(default_factory=dict)  # {support: 0.x, refute: 0.x, neutral: 0.x}
    repaired: bool = False       # 是否经过修复
    repair_iteration: int = 0


@dataclass
class VerificationReport:
    """完整核验报告"""
    question: str
    original_answer: str
    sentences: list[SentenceVerification]
    final_answer: Optional[str]
    status: AgentStatus
    iterations: int
    total_time_ms: float
    refuse_reason: Optional[str] = None

    @property
    def supported_ratio(self) -> float:
        if not self.sentences:
            return 0.0
        supported = sum(1 for s in self.sentences if s.label == EvidenceLabel.SUPPORT)
        return supported / len(self.sentences)

    @property
    def hallucination_stats(self) -> dict:
        stats = {c.value: 0 for c in HallucinationCause}
        for s in self.sentences:
            if s.cause:
                stats[s.cause.value] += 1
        return stats

    def to_traceable_output(self) -> dict:
        """生成可追溯输出，每句绑定原文证据"""
        result = {
            "answer": self.final_answer,
            "status": self.status.value,
            "supported_ratio": round(self.supported_ratio, 3),
            "iterations": self.iterations,
            "time_ms": round(self.total_time_ms, 1),
            "sentence_details": [],
        }
        for sv in self.sentences:
            detail = {
                "sentence": sv.sentence,
                "label": sv.label.value,
                "cause": sv.cause.value if sv.cause else None,
                "rule_check": sv.rule_check_detail or "pass",
                "sources": [
                    {
                        "source": c.source,
                        "page": c.page,
                        "excerpt": c.content[:120] + "..." if len(c.content) > 120 else c.content,
                        "metadata": c.metadata,
                    }
                    for c in sv.supporting_chunks
                ],
            }
            result["sentence_details"].append(detail)
        if self.refuse_reason:
            result["refuse_reason"] = self.refuse_reason
        return result
