"""细粒度核验引擎
两层级联：
  1. 规则硬校验（实体、数值、型号版本）→ 快速排除明显错误
  2. NLI 语义蕴含三分类 → 细粒度判定支持/反驳/不足
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from ..models import Chunk, EvidenceLabel, HallucinationCause, SentenceVerification
from ..config import get_config


# ── 句子分割 ──────────────────────────────────────────────────────────

_SENT_BOUNDARY = re.compile(r"(?<=[。！？!?])\s*|(?<=\.)(?<!\d\.)\s*")


def split_sentences(text: str) -> list[str]:
    """将回答切分为可独立核验的短句"""
    raw = _SENT_BOUNDARY.split(text.strip())
    sentences = []
    for s in raw:
        s = s.strip()
        if len(s) < 5:  # 过滤极短片段
            continue
        # 长句进一步按逗号/分号拆分（保留上下文语义）
        if len(s) > 120:
            sub = re.split(r"[，；,;]", s)
            sentences.extend([x.strip() for x in sub if len(x.strip()) > 5])
        else:
            sentences.append(s)
    return sentences or [text]


# ── 元话语句识别（引用来源、自我评价、补充说明等，不参与事实核验）────────

_META_PATTERN = re.compile(
    r"^[（(【\[]?\s*(依据来源|资料来源|引用来源|数据来源|参考来源|注\s*[:：]|备注|说明\s*[:：]|本回答|以上回答|综上)"
    r"|参考文档片段|文档片段\s*\[?\d|片段\s*\[\d+\]"
    r"|具备明确的事实支撑|符合.{0,12}要求"
)


def _is_meta_sentence(sentence: str) -> bool:
    return bool(_META_PATTERN.search(sentence))


# ── 规则硬校验 ────────────────────────────────────────────────────────

_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?(?:\s*[%％°℃kKmMgGwW])?")
_VERSION_IN_SENT = re.compile(r"[vV]?\d+\.\d+[a-zA-Z]?|\b[A-Z]{1,3}\d{2,6}\b")


@dataclass
class RuleCheckResult:
    passed: bool
    cause: Optional[HallucinationCause] = None
    detail: str = ""


def rule_check(sentence: str, chunks: list[Chunk]) -> RuleCheckResult:
    """
    规则硬校验：
    - 数值实体匹配
    - 型号/版本号校验
    返回未通过时的初步成因判定
    """
    if not chunks:
        return RuleCheckResult(False, HallucinationCause.RETRIEVAL_MISS, "无可用证据片段")

    context_text = " ".join(c.content for c in chunks)

    # 1. 版本/型号号检查
    sent_versions = set(_VERSION_IN_SENT.findall(sentence))
    if sent_versions:
        matched = any(v in context_text for v in sent_versions)
        if not matched:
            # 判断是版本不匹配还是完全未命中
            # 若文档中有任意版本号出现，则归类为 version_mismatch
            doc_versions = set(_VERSION_IN_SENT.findall(context_text))
            if doc_versions:
                return RuleCheckResult(
                    False,
                    HallucinationCause.VERSION_MISMATCH,
                    f"句中版本/型号 {sent_versions} 未见于证据片段（文档版本：{doc_versions}）",
                )
            else:
                return RuleCheckResult(
                    False,
                    HallucinationCause.RETRIEVAL_MISS,
                    f"句中版本/型号 {sent_versions} 未在文档中出现",
                )

    # 2. 关键数值检查
    sent_numbers = set(_NUMBER_PATTERN.findall(sentence))
    for num in sent_numbers:
        if num.isdigit() and len(num) <= 2:
            continue  # 忽略普通序号
        if num not in context_text:
            return RuleCheckResult(
                False,
                HallucinationCause.EVIDENCE_INSUFFICIENT,
                f"数值 '{num}' 未在证据片段中找到",
            )

    return RuleCheckResult(True, detail="规则校验通过")


# ── NLI 三分类模型 ────────────────────────────────────────────────────

class NLIModel:
    """自然语言推理（蕴含关系）三分类：支持/反驳/中性"""
    _instance = None

    def __init__(self):
        cfg = get_config()["nli"]
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        logger.info(f"Loading NLI model: {cfg['model']}")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg["model"])
        self.model.eval()
        self.device = "cuda" if cfg.get("device") == "cuda" else "cpu"
        self.model.to(self.device)
        # id2label: entailment / neutral / contradiction
        self._label_ids = {}
        for idx, name in self.model.config.id2label.items():
            name = name.lower()
            if "entail" in name:
                self._label_ids["support"] = idx
            elif "contradict" in name:
                self._label_ids["refute"] = idx
            else:
                self._label_ids["insufficient"] = idx
        self.threshold_support = cfg.get("threshold_support", 0.6)
        self.threshold_refute = cfg.get("threshold_refute", 0.6)

    @classmethod
    def get(cls) -> "NLIModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def classify(self, hypothesis: str, premise: str) -> dict:
        """
        Returns: {"support": float, "refute": float, "insufficient": float}
        premise=证据文本, hypothesis=待核验句子
        """
        import torch
        try:
            enc = self.tokenizer(
                premise,
                hypothesis,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits[0]
            probs = torch.softmax(logits, dim=-1).tolist()
            return {
                "support": probs[self._label_ids["support"]],
                "refute": probs[self._label_ids["refute"]],
                "insufficient": probs[self._label_ids["insufficient"]],
            }
        except Exception as e:
            logger.warning(f"NLI classify error: {e}, defaulting to insufficient")
            return {"support": 0.0, "refute": 0.0, "insufficient": 1.0}

    def label_from_scores(self, scores: dict) -> EvidenceLabel:
        if scores.get("support", 0) >= self.threshold_support:
            return EvidenceLabel.SUPPORT
        if scores.get("refute", 0) >= self.threshold_refute:
            return EvidenceLabel.REFUTE
        return EvidenceLabel.INSUFFICIENT


# ── 级联核验器 ────────────────────────────────────────────────────────

class CascadeVerifier:
    """
    级联核验流程：
      sentence → rule_check → (pass) → NLI 三分类
                            → (fail) → 直接判定成因
    """

    def __init__(self, use_nli: bool = True):
        self.use_nli = use_nli
        self._nli: Optional[NLIModel] = None

    def _get_nli(self) -> NLIModel:
        if self._nli is None:
            self._nli = NLIModel.get()
        return self._nli

    def verify_sentence(self, sentence: str, evidence_chunks: list[Chunk]) -> SentenceVerification:
        sv = SentenceVerification(sentence=sentence, label=EvidenceLabel.INSUFFICIENT)

        # 第一层：规则硬校验
        rule_result = rule_check(sentence, evidence_chunks)
        sv.rule_check_passed = rule_result.passed
        sv.rule_check_detail = rule_result.detail

        if not rule_result.passed:
            sv.label = EvidenceLabel.INSUFFICIENT
            sv.cause = rule_result.cause
            sv.supporting_chunks = evidence_chunks[:2] if evidence_chunks else []
            return sv

        # 第二层：NLI 语义蕴含
        if not self.use_nli or not evidence_chunks:
            # 无 NLI 时，规则通过即视为支持（降级策略）
            sv.label = EvidenceLabel.SUPPORT if evidence_chunks else EvidenceLabel.INSUFFICIENT
            sv.cause = None if evidence_chunks else HallucinationCause.RETRIEVAL_MISS
            sv.supporting_chunks = evidence_chunks
            return sv

        # 用最相关的前3个 chunk 做 NLI，取最高支持分
        best_label = EvidenceLabel.INSUFFICIENT
        best_scores = {"support": 0.0, "refute": 0.0, "insufficient": 1.0}
        best_chunks = []

        nli_model = self._get_nli()
        for chunk in evidence_chunks[:3]:
            scores = nli_model.classify(sentence, chunk.content)
            label = nli_model.label_from_scores(scores)
            if label == EvidenceLabel.SUPPORT:
                best_label = EvidenceLabel.SUPPORT
                best_scores = scores
                best_chunks = [chunk]
                break
            if scores.get("support", 0) > best_scores.get("support", 0):
                best_scores = scores
                best_chunks = [chunk]
                best_label = label

        sv.label = best_label
        sv.nli_scores = best_scores
        sv.supporting_chunks = best_chunks if best_label == EvidenceLabel.SUPPORT else evidence_chunks[:2]

        # 成因推断
        if best_label != EvidenceLabel.SUPPORT:
            if best_label == EvidenceLabel.REFUTE:
                sv.cause = HallucinationCause.EVIDENCE_INSUFFICIENT
            elif not evidence_chunks:
                sv.cause = HallucinationCause.RETRIEVAL_MISS
            else:
                sv.cause = HallucinationCause.EVIDENCE_INSUFFICIENT

        return sv

    def verify_answer(self, answer: str, evidence_chunks: list[Chunk]) -> list[SentenceVerification]:
        """对完整回答做句子级批量核验（仅核验事实陈述，跳过引用/自评等元话语）"""
        sentences = split_sentences(answer)
        factual = [s for s in sentences if not _is_meta_sentence(s)]
        if not factual:  # 全为元话语时保留原句，避免空核验
            factual = sentences
        results = []
        for sent in factual:
            sv = self.verify_sentence(sent, evidence_chunks)
            results.append(sv)
        return results
