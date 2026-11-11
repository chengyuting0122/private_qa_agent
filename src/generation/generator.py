"""内层生成智能体：基于检索证据生成回答初稿"""
from __future__ import annotations
from typing import Optional

import httpx
from loguru import logger

from ..models import Chunk
from ..config import get_config


SYSTEM_PROMPT = """你是一个严谨的私域文档问答助手。
回答要求：
1. 仅依据提供的参考文档片段作答，不得凭空捏造
2. 每个事实性陈述都应能在参考片段中找到依据
3. 如参考片段不足以完整回答问题，明确说明哪些部分无法确认
4. 使用准确、专业的表述，避免模糊化措辞
5. 回答用中文，结构清晰"""

QUERY_TEMPLATE = """参考文档片段（共{n}条）：

{context}

---
用户问题：{question}

请基于以上参考片段，给出完整、准确的回答："""


def build_context(chunks: list[Chunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta_str = ""
        if chunk.metadata.get("version"):
            meta_str += f" [版本:{chunk.metadata['version']}]"
        if chunk.metadata.get("model_id"):
            meta_str += f" [型号:{chunk.metadata['model_id']}]"
        src = chunk.source.split("/")[-1] if chunk.source else "未知"
        page_str = f" 第{chunk.page}页" if chunk.page else ""
        header = f"[{i}] {src}{page_str}{meta_str}"
        parts.append(f"{header}\n{chunk.content}")
    return "\n\n".join(parts)


class GenerationAgent:
    def __init__(self):
        self.cfg = get_config()["llm"]

    def generate(self, question: str, chunks: list[Chunk]) -> str:
        context = build_context(chunks)
        prompt = QUERY_TEMPLATE.format(n=len(chunks), context=context, question=question)

        provider = self.cfg.get("provider", "ollama")
        if provider == "ollama":
            return self._call_ollama(prompt)
        else:
            return self._call_openai_compatible(prompt)

    def _call_ollama(self, prompt: str) -> str:
        url = f"{self.cfg['base_url']}/api/generate"
        payload = {
            "model": self.cfg.get("model", "qwen2.5:7b"),
            "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
            "stream": False,
            "options": {
                "temperature": self.cfg.get("temperature", 0.1),
                "num_predict": self.cfg.get("max_tokens", 2048),
            },
        }
        try:
            resp = httpx.post(url, json=payload, timeout=120.0)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            raise RuntimeError(f"LLM call failed: {e}") from e

    def _call_openai_compatible(self, prompt: str) -> str:
        url = f"{self.cfg['base_url']}/v1/chat/completions"
        payload = {
            "model": self.cfg.get("model", "gpt-3.5-turbo"),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.cfg.get("temperature", 0.1),
            "max_tokens": self.cfg.get("max_tokens", 2048),
        }
        headers = {}
        if self.cfg.get("api_key"):
            headers["Authorization"] = f"Bearer {self.cfg['api_key']}"
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=120.0)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"OpenAI compatible call failed: {e}")
            raise RuntimeError(f"LLM call failed: {e}") from e

    def refine(self, question: str, original_answer: str, chunks: list[Chunk], failed_sentences: list[str]) -> str:
        """修复后重新生成（聚焦失败句子）"""
        context = build_context(chunks)
        failed_str = "\n".join(f"- {s}" for s in failed_sentences)
        prompt = f"""参考文档片段（共{len(chunks)}条）：

{context}

---
原始回答：
{original_answer}

以下句子缺乏证据支撑，请基于新增参考片段进行修正或删除：
{failed_str}

用户原始问题：{question}

请给出修正后的完整回答："""

        provider = self.cfg.get("provider", "ollama")
        if provider == "ollama":
            return self._call_ollama(prompt)
        else:
            return self._call_openai_compatible(prompt)
