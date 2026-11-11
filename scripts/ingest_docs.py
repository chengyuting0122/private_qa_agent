"""文档摄入脚本"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.retrieval.retriever import VectorStore
from src.ingestion.parser import ingest_directory


def main():
    parser = argparse.ArgumentParser(description="导入文档到向量库")
    parser.add_argument("--docs", default="data/sample_docs", help="文档目录")
    parser.add_argument("--config", default=None, help="配置文件路径")
    args = parser.parse_args()

    if args.config:
        from src.config import load_config
        import src.config as cfg_module
        cfg_module._config = load_config(args.config)

    print(f"正在导入文档目录: {args.docs}")
    vs = VectorStore()
    count = ingest_directory(args.docs, vs)
    print(f"✅ 导入完成，共写入 {count} 个文本块")
    print(f"   向量库当前共有 {vs.count()} 个文本块")


if __name__ == "__main__":
    main()
