"""单元测试：无需 LLM/NLI，纯逻辑验证"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.models import Chunk, EvidenceLabel, HallucinationCause
from src.verification.verifier import split_sentences, rule_check, CascadeVerifier
from src.ingestion.parser import sliding_chunks, extract_metadata


# ── 句子分割测试 ──────────────────────────────────────────────────────

def test_split_sentences_basic():
    text = "额定推力为3200N。最大推力为4100N。工作温度为-40°C至65°C。"
    sents = split_sentences(text)
    assert len(sents) == 3
    assert "3200N" in sents[0]


def test_split_sentences_long():
    text = "这是一个很长的句子，包含很多信息，关于发动机型号XJ-3000，版本V1.2，额定推力3200N，最大推力4100N，工作温度范围为-40摄氏度到65摄氏度，燃油消耗率0.38。"
    sents = split_sentences(text)
    assert len(sents) >= 1  # 至少分出一段


# ── 规则硬校验测试 ────────────────────────────────────────────────────

def make_chunk(content: str, metadata: dict = None) -> Chunk:
    return Chunk(
        chunk_id="test",
        content=content,
        source="test.txt",
        metadata=metadata or {},
    )


def test_rule_check_no_chunks():
    result = rule_check("额定推力为3200N", [])
    assert not result.passed
    assert result.cause == HallucinationCause.RETRIEVAL_MISS


def test_rule_check_number_match():
    chunk = make_chunk("额定推力：3200 N，最大推力：4100 N")
    result = rule_check("额定推力为3200N", [chunk])
    assert result.passed


def test_rule_check_number_mismatch():
    chunk = make_chunk("额定推力：3200 N，最大推力：4100 N")
    result = rule_check("额定推力为9999N", [chunk])
    assert not result.passed
    assert result.cause == HallucinationCause.EVIDENCE_INSUFFICIENT


def test_rule_check_version_mismatch():
    chunk = make_chunk("版本V1.2 额定推力3200N", {"version": "V1.2"})
    result = rule_check("V2.0版本的额定推力为3456N", [chunk])
    assert not result.passed
    assert result.cause == HallucinationCause.VERSION_MISMATCH


def test_rule_check_version_match():
    chunk = make_chunk("V1.2版本 额定推力3200N")
    result = rule_check("V1.2版本的额定推力为3200N", [chunk])
    assert result.passed


# ── 分块测试 ──────────────────────────────────────────────────────────

def test_sliding_chunks_basic():
    text = "一" * 1000
    chunks = sliding_chunks(text, chunk_size=400, overlap=50)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 400 + 10  # 允许小误差


def test_sliding_chunks_short():
    text = "这是短文本。"
    chunks = sliding_chunks(text, chunk_size=400)
    assert len(chunks) == 1
    assert chunks[0] == text.strip()


# ── 元数据提取测试 ────────────────────────────────────────────────────

def test_extract_metadata_version():
    text = "版本：V2.0\n发动机手册"
    meta = extract_metadata(text, "test.txt")
    assert meta.get("version") == "2.0"


def test_extract_metadata_model():
    text = "型号：XJ-3000\n发动机"
    meta = extract_metadata(text, "test.txt")
    assert meta.get("model_id") == "XJ-3000"


# ── CascadeVerifier（无 NLI 模式）────────────────────────────────────

def test_verifier_no_nli_support():
    verifier = CascadeVerifier(use_nli=False)
    chunk = make_chunk("额定推力3200N，最大推力4100N")
    sv = verifier.verify_sentence("额定推力为3200N", [chunk])
    assert sv.label == EvidenceLabel.SUPPORT


def test_verifier_no_nli_retrieval_miss():
    verifier = CascadeVerifier(use_nli=False)
    sv = verifier.verify_sentence("额定推力为3200N", [])
    assert sv.label == EvidenceLabel.INSUFFICIENT
    assert sv.cause == HallucinationCause.RETRIEVAL_MISS


def test_verifier_full_answer():
    verifier = CascadeVerifier(use_nli=False)
    chunk = make_chunk("额定推力3200N。工作温度-40°C至65°C。")
    answer = "额定推力为3200N。工作温度范围是-40°C至65°C。"
    svs = verifier.verify_answer(answer, [chunk])
    assert len(svs) >= 1


# ── 数据模型测试 ──────────────────────────────────────────────────────

def test_verification_report_supported_ratio():
    from src.models import VerificationReport, AgentStatus, SentenceVerification
    svs = [
        SentenceVerification("句1", EvidenceLabel.SUPPORT),
        SentenceVerification("句2", EvidenceLabel.SUPPORT),
        SentenceVerification("句3", EvidenceLabel.INSUFFICIENT),
        SentenceVerification("句4", EvidenceLabel.INSUFFICIENT),
    ]
    report = VerificationReport(
        question="test",
        original_answer="test",
        sentences=svs,
        final_answer="test",
        status=AgentStatus.PARTIAL,
        iterations=1,
        total_time_ms=100.0,
    )
    assert report.supported_ratio == 0.5


def test_traceable_output_structure():
    from src.models import VerificationReport, AgentStatus, SentenceVerification
    svs = [SentenceVerification("测试句子", EvidenceLabel.SUPPORT)]
    report = VerificationReport(
        question="问题",
        original_answer="回答",
        sentences=svs,
        final_answer="回答",
        status=AgentStatus.SUCCESS,
        iterations=0,
        total_time_ms=50.0,
    )
    out = report.to_traceable_output()
    assert "answer" in out
    assert "sentence_details" in out
    assert out["supported_ratio"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
