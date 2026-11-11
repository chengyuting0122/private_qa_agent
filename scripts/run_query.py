"""命令行问答测试（不依赖 LLM，使用 mock 模式验证核验流程）"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_mock_query(question: str, verbose: bool = False):
    """
    Mock 模式：不调用真实 LLM，直接用检索到的 chunk 拼接回答，
    用于验证核验、修复流程是否正常工作。
    """
    from src.retrieval.retriever import VectorStore, HybridRetriever
    from src.verification.verifier import CascadeVerifier, split_sentences
    from src.models import Chunk, EvidenceLabel

    vs = VectorStore()
    if vs.count() == 0:
        print("❌ 向量库为空，请先运行: python scripts/ingest_docs.py")
        return

    retriever = HybridRetriever(vs)
    verifier = CascadeVerifier(use_nli=False)  # mock 模式不加载 NLI 模型

    chunks = retriever.retrieve(question)
    if not chunks:
        print("❌ 未找到相关文档片段")
        return

    print(f"\n📄 检索到 {len(chunks)} 个文本块\n")

    # 用检索到的片段直接作为"回答"（测试核验流程）
    mock_answer = " ".join(c.content[:100] for c in chunks[:2])

    print(f"问题: {question}")
    print(f"\n📝 Mock 回答（截取）:\n{mock_answer[:200]}...\n")
    print("─" * 60)

    svs = verifier.verify_answer(mock_answer, chunks)

    label_icon = {EvidenceLabel.SUPPORT: "✅", EvidenceLabel.REFUTE: "❌", EvidenceLabel.INSUFFICIENT: "⚠️"}
    supported = sum(1 for sv in svs if sv.label == EvidenceLabel.SUPPORT)

    for i, sv in enumerate(svs, 1):
        icon = label_icon.get(sv.label, "?")
        cause_str = f" [{sv.cause.value}]" if sv.cause else ""
        print(f"{icon} 句{i}: {sv.sentence[:60]}...{cause_str}")
        if verbose and sv.supporting_chunks:
            src = sv.supporting_chunks[0]
            print(f"     来源: {src.source.split('/')[-1]} | 规则: {sv.rule_check_detail}")

    print(f"\n支持率: {supported}/{len(svs)} = {supported/max(len(svs),1)*100:.0f}%")

    if verbose:
        print("\n📋 JSON 详情:")
        details = []
        for sv in svs:
            details.append({
                "sentence": sv.sentence,
                "label": sv.label.value,
                "cause": sv.cause.value if sv.cause else None,
                "rule_check": sv.rule_check_detail,
            })
        print(json.dumps(details, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="命令行问答测试")
    parser.add_argument("--question", "-q", required=True, help="问题")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")
    args = parser.parse_args()
    run_mock_query(args.question, args.verbose)


if __name__ == "__main__":
    main()
