"""文档摄入模块：解析 → 分块 → 元数据提取 → 入库"""
from __future__ import annotations
import re
import uuid
import hashlib
from pathlib import Path
from typing import Iterator

from loguru import logger

from ..models import Chunk
from ..config import get_config


# ── 元数据提取：型号、版本、章节 ──────────────────────────────────────

_VERSION_PATTERN = re.compile(
    r"(?:版本|version|ver\.?|v)[：:\s]*([0-9]+(?:\.[0-9]+)*[a-zA-Z]?)",
    re.IGNORECASE,
)
_MODEL_PATTERN = re.compile(
    r"(?:型号|model|编号)[：:\s]*([A-Z0-9\-]{2,20})",
    re.IGNORECASE,
)
_CHAPTER_PATTERN = re.compile(r"^第[一二三四五六七八九十\d]+[章节条款]|^\d+(\.\d+)+\s+\S")


def extract_metadata(text: str, filename: str = "") -> dict:
    meta = {"filename": filename}
    ver = _VERSION_PATTERN.search(text[:500])
    if ver:
        meta["version"] = ver.group(1)
    model = _MODEL_PATTERN.search(text[:500])
    if model:
        meta["model_id"] = model.group(1)
    return meta


# ── 文本分块 ──────────────────────────────────────────────────────────

def sliding_chunks(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """按字符滑动窗口分块，在句号/换行处切割避免截断句子"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # 向后找最近的句子边界
        if end < len(text):
            for sep in ("。", "\n", "；", "."):
                pos = text.rfind(sep, start, end)
                if pos != -1 and pos > start + chunk_size // 2:
                    end = pos + 1
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        next_start = end - overlap
        if next_start <= start:  # 防止无限循环（末尾短块）
            next_start = end
        start = next_start
    return chunks


# ── 文件解析器 ────────────────────────────────────────────────────────

def _parse_txt(path: Path) -> list[tuple[str, dict]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    meta = extract_metadata(text, path.name)
    return [(text, meta)]


def _parse_pdf(path: Path) -> list[tuple[str, dict]]:
    try:
        import PyPDF2
        pages = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append((text, {"page": i + 1}))
        full_text = "\n".join(t for t, _ in pages)
        global_meta = extract_metadata(full_text, path.name)
        return [(t, {**global_meta, **m}) for t, m in pages]
    except ImportError:
        logger.warning("PyPDF2 not installed, treating PDF as text")
        return _parse_txt(path)
    except Exception as e:
        logger.error(f"PDF parse error {path}: {e}")
        return []


def _parse_docx(path: Path) -> list[tuple[str, dict]]:
    try:
        from docx import Document
        doc = Document(path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        meta = extract_metadata(text, path.name)
        return [(text, meta)]
    except ImportError:
        logger.warning("python-docx not installed")
        return []
    except Exception as e:
        logger.error(f"DOCX parse error {path}: {e}")
        return []


_PARSERS = {
    ".txt": _parse_txt,
    ".md": _parse_txt,
    ".pdf": _parse_pdf,
    ".docx": _parse_docx,
}


def parse_file(path: Path) -> list[tuple[str, dict]]:
    suffix = path.suffix.lower()
    parser = _PARSERS.get(suffix, _parse_txt)
    return parser(path)


# ── 主流程 ────────────────────────────────────────────────────────────

def ingest_directory(doc_dir: str, vector_store) -> int:
    """将目录下所有文档摄入向量库，返回写入 chunk 数量"""
    cfg = get_config()["ingestion"]
    chunk_size = cfg.get("chunk_size", 400)
    overlap = cfg.get("chunk_overlap", 50)

    doc_path = Path(doc_dir)
    total = 0

    for fpath in sorted(doc_path.rglob("*")):
        if not fpath.is_file():
            continue
        if fpath.suffix.lower() not in _PARSERS:
            continue

        logger.info(f"Ingesting: {fpath.name}")
        sections = parse_file(fpath)

        chunks: list[Chunk] = []
        for text, meta in sections:
            for fragment in sliding_chunks(text, chunk_size, overlap):
                cid = hashlib.md5(fragment.encode()).hexdigest()[:12]
                chunk = Chunk(
                    chunk_id=cid,
                    content=fragment,
                    source=str(fpath),
                    page=meta.get("page"),
                    metadata={k: v for k, v in meta.items() if k != "page"},
                )
                chunks.append(chunk)

        if chunks:
            vector_store.add_chunks(chunks)
            total += len(chunks)
            logger.info(f"  → {len(chunks)} chunks")

    logger.info(f"Ingestion complete: {total} total chunks")
    return total
