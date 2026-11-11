"""配置加载模块"""
import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


def load_config(path: Optional[str] = None) -> dict:
    cfg_path = Path(path) if path else _CONFIG_PATH
    if not cfg_path.exists():
        example = cfg_path.parent / "config.yaml.example"
        if example.exists():
            cfg_path = example
        else:
            return _default_config()
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _default_config() -> dict:
    return {
        "llm": {
            "provider": "vllm",
            "base_url": "http://192.168.31.40:18004",
            "model": "Qwen3-4B-Instruct-2507",
            "temperature": 0.1,
            "max_tokens": 2048,
        },
        "embedding": {
            "provider": "vllm",
            "base_url": "http://192.168.31.40:18005",
            "model": "qwen3-embedding",
            "batch_size": 32,
        },
        "nli": {
            "model": "./models/mdeberta-xnli",
            "device": "cpu",
            "threshold_support": 0.6,
            "threshold_refute": 0.6,
        },
        "vectordb": {
            "type": "chroma",
            "persist_dir": "./data/chroma_db",
            "collection": "private_docs",
        },
        "retrieval": {
            "top_k": 5,
            "expand_k": 10,
            "use_bm25": True,
            "rerank": False,
        },
        "ingestion": {
            "chunk_size": 400,
            "chunk_overlap": 50,
            "extract_metadata": True,
        },
        "agent": {
            "max_iterations": 3,
            "refuse_on_exhaustion": True,
            "min_supported_ratio": 0.7,
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
        },
    }


# 全局配置单例
_config = None


def get_config() -> dict:
    global _config
    if _config is None:
        _config = load_config()
    return _config
