"""轻量单元测试：零重型依赖（无 numpy/torch/transformers），纯逻辑验证"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 注意：直接 import 模块文件，不走包级 __init__，避免触发懒加载之外的依赖
import importlib, types

# ── 手动 stub 掉重型依赖，让模块可以 import ────────────────────────────
for mod in ["numpy", "loguru", "sentence_transformers", "chromadb",
            "transformers", "torch", "rank_bm25", "jieba", "sklearn"]:
    if mod not in sys.modules:
        stub = types.ModuleType(mod)
        # loguru.logger stub
        if mod == "loguru":
            class _Logger:
                def info(self, *a, **k): pass
                def debug(self, *a, **k): pass
                def warning(self, *a, **k): pass
                def error(self, *a, **k): pass
                def exception(self, *a, **k): pass
            stub.logger = _Logger()
        sys.modules[mod] = stub

# ── 现在可以安全 import ─────────────────────────────────────────────────
from src.models import (
    Chunk, EvidenceLabel, HallucinationCause,
    SentenceVerification, VerificationReport, AgentStatus,
)
from src.verification.verifier import split_sentences, rule_check, CascadeVerifier
from src.ingestion.parser import sliding_chunks, extract_metadata


# ══════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════
def make_chunk(content: str, metadata: dict = None) -> Chunk:
    return Chunk(chunk_id="t", content=content, source="test.txt", metadata=metadata or {})


def run(name, fn):
    try:
        fn()
        print(f"  ✅  {name}")
        return True
    except AssertionError as e:
        print(f"  ❌  {name}  → AssertionError: {e}")
        return False
    except Exception as e:
        print(f"  💥  {name}  → {type(e).__name__}: {e}")
        return False


# ══════════════════════════════════════════════
# 测试用例
# ══════════════════════════════════════════════

def t_split_basic():
    sents = split_sentences("额定推力为3200N。最大推力为4100N。工作温度为-40°C至65°C。")
    assert len(sents) == 3, f"got {len(sents)}"
    assert "3200N" in sents[0]

def t_split_single():
    sents = split_sentences("一句话没有标点")
    assert len(sents) == 1

def t_chunk_basic():
    text = "一" * 1000
    chunks = sliding_chunks(text, 400, 50)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 410, f"chunk too long: {len(c)}"

def t_chunk_short():
    chunks = sliding_chunks("短文本。", 400)
    assert chunks[0].strip() == "短文本。"

def t_meta_version():
    meta = extract_metadata("版本：V2.0\n发动机手册", "f.txt")
    assert meta.get("version") == "2.0", f"got {meta}"

def t_meta_model():
    meta = extract_metadata("型号：XJ-3000\n发动机", "f.txt")
    assert meta.get("model_id") == "XJ-3000", f"got {meta}"

def t_rule_no_chunks():
    r = rule_check("额定推力为3200N", [])
    assert not r.passed
    assert r.cause == HallucinationCause.RETRIEVAL_MISS

def t_rule_number_match():
    r = rule_check("额定推力为3200N", [make_chunk("额定推力：3200 N，最大推力：4100 N")])
    assert r.passed, r.detail

def t_rule_number_mismatch():
    r = rule_check("额定推力为9999N", [make_chunk("额定推力：3200 N")])
    assert not r.passed
    assert r.cause == HallucinationCause.EVIDENCE_INSUFFICIENT

def t_rule_version_mismatch():
    r = rule_check("V2.0版本的额定推力为3456N", [make_chunk("V1.2版本 额定推力3200N")])
    assert not r.passed
    assert r.cause == HallucinationCause.VERSION_MISMATCH, f"got {r.cause}"

def t_rule_version_match():
    r = rule_check("V1.2版本的额定推力为3200N", [make_chunk("V1.2版本 额定推力3200N")])
    assert r.passed, r.detail

def t_verifier_support():
    v = CascadeVerifier(use_nli=False)
    sv = v.verify_sentence("额定推力为3200N", [make_chunk("额定推力3200N，最大推力4100N")])
    assert sv.label == EvidenceLabel.SUPPORT, f"got {sv.label}"

def t_verifier_miss():
    v = CascadeVerifier(use_nli=False)
    sv = v.verify_sentence("额定推力为3200N", [])
    assert sv.label == EvidenceLabel.INSUFFICIENT
    assert sv.cause == HallucinationCause.RETRIEVAL_MISS

def t_verifier_full_answer():
    v = CascadeVerifier(use_nli=False)
    c = make_chunk("额定推力3200N。工作温度-40°C至65°C。")
    svs = v.verify_answer("额定推力为3200N。工作温度范围是-40°C至65°C。", [c])
    assert len(svs) >= 1

def t_report_ratio():
    svs = [
        SentenceVerification("句1", EvidenceLabel.SUPPORT),
        SentenceVerification("句2", EvidenceLabel.SUPPORT),
        SentenceVerification("句3", EvidenceLabel.INSUFFICIENT),
        SentenceVerification("句4", EvidenceLabel.INSUFFICIENT),
    ]
    r = VerificationReport("q", "a", svs, "a", AgentStatus.PARTIAL, 1, 100.0)
    assert r.supported_ratio == 0.5, f"got {r.supported_ratio}"

def t_report_traceable():
    svs = [SentenceVerification("测试句子", EvidenceLabel.SUPPORT)]
    r = VerificationReport("问题", "回答", svs, "回答", AgentStatus.SUCCESS, 0, 50.0)
    out = r.to_traceable_output()
    assert "answer" in out
    assert "sentence_details" in out
    assert out["supported_ratio"] == 1.0

def t_report_refuse():
    r = VerificationReport("q", "a", [], None, AgentStatus.REFUSED, 3, 200.0,
                            refuse_reason="证据不足")
    out = r.to_traceable_output()
    assert out["status"] == "refused"
    assert "refuse_reason" in out

def t_hallucination_stats():
    svs = [
        SentenceVerification("句1", EvidenceLabel.INSUFFICIENT,
                             cause=HallucinationCause.RETRIEVAL_MISS),
        SentenceVerification("句2", EvidenceLabel.INSUFFICIENT,
                             cause=HallucinationCause.VERSION_MISMATCH),
        SentenceVerification("句3", EvidenceLabel.SUPPORT),
    ]
    r = VerificationReport("q", "a", svs, "a", AgentStatus.PARTIAL, 1, 100.0)
    stats = r.hallucination_stats
    assert stats["retrieval_miss"] == 1
    assert stats["version_mismatch"] == 1
    assert stats["evidence_insufficient"] == 0


# ══════════════════════════════════════════════
# 主运行
# ══════════════════════════════════════════════

TESTS = [
    ("句子分割-基础", t_split_basic),
    ("句子分割-无标点", t_split_single),
    ("滑动分块-基础", t_chunk_basic),
    ("滑动分块-短文", t_chunk_short),
    ("元数据-版本提取", t_meta_version),
    ("元数据-型号提取", t_meta_model),
    ("规则校验-无片段", t_rule_no_chunks),
    ("规则校验-数值匹配", t_rule_number_match),
    ("规则校验-数值不匹配", t_rule_number_mismatch),
    ("规则校验-版本不匹配", t_rule_version_mismatch),
    ("规则校验-版本匹配", t_rule_version_match),
    ("核验器-支持", t_verifier_support),
    ("核验器-未命中", t_verifier_miss),
    ("核验器-完整回答", t_verifier_full_answer),
    ("报告-支持率计算", t_report_ratio),
    ("报告-可追溯输出", t_report_traceable),
    ("报告-拒答输出", t_report_refuse),
    ("报告-幻觉统计", t_hallucination_stats),
]

if __name__ == "__main__":
    print("\n私域可信问答智能体 — 核心逻辑单元测试\n" + "─"*50)
    passed = sum(run(name, fn) for name, fn in TESTS)
    total = len(TESTS)
    print(f"\n{'─'*50}")
    print(f"结果: {passed}/{total} 通过" + ("  🎉" if passed == total else "  ⚠️ 有失败项"))
