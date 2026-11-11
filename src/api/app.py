"""FastAPI 接口层 + 简单前端"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from loguru import logger

from ..config import get_config
from ..retrieval.retriever import VectorStore
from ..agent.orchestrator import TrustQAAgent


# ── 初始化 ────────────────────────────────────────────────────────────

app = FastAPI(
    title="私域可信问答智能体",
    description="基于细粒度证据核验的私域文档问答系统",
    version="1.0.0",
)

cfg = get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.get("api", {}).get("cors_origins", ["*"]),
    allow_methods=["*"],
    allow_headers=["*"],
)

_vector_store: Optional[VectorStore] = None
_agent: Optional[TrustQAAgent] = None
_agent_use_nli: Optional[bool] = None


def get_agent(use_nli: bool = False) -> TrustQAAgent:
    global _vector_store, _agent, _agent_use_nli
    if _vector_store is None:
        _vector_store = VectorStore()
    if _agent is None or _agent_use_nli != use_nli:
        _agent = TrustQAAgent(_vector_store, use_nli=use_nli)
        _agent_use_nli = use_nli
    return _agent


# ── 请求/响应模型 ─────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    use_nli: bool = False


class IngestRequest(BaseModel):
    doc_dir: str = "./data/sample_docs"


# ── 接口 ──────────────────────────────────────────────────────────────

@app.post("/query")
async def query(req: QueryRequest):
    agent = get_agent(req.use_nli)
    try:
        report = agent.query(req.question)
        return report.to_traceable_output()
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest(req: IngestRequest):
    from ..ingestion.parser import ingest_directory
    vs = VectorStore()
    count = ingest_directory(req.doc_dir, vs)
    return {"status": "ok", "chunks_ingested": count}


@app.get("/stats")
async def stats():
    agent = get_agent()
    return {
        "chunks_in_db": agent.retriever.vector_store.count(),
        "config": {
            "max_iterations": agent.max_iterations,
            "min_supported_ratio": agent.min_supported_ratio,
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── 前端演示页面 ──────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


HTML_PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>私域可信问答智能体</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f1117; color: #e2e8f0; min-height: 100vh; }
  .container { max-width: 960px; margin: 0 auto; padding: 2rem; }
  h1 { font-size: 1.5rem; font-weight: 700; color: #7c3aed; margin-bottom: 0.5rem; }
  .subtitle { font-size: 0.85rem; color: #64748b; margin-bottom: 2rem; }
  .card { background: #1e2130; border: 1px solid #2d3748; border-radius: 12px;
          padding: 1.5rem; margin-bottom: 1.5rem; }
  textarea { width: 100%; background: #0f1117; border: 1px solid #374151;
             border-radius: 8px; color: #e2e8f0; padding: 0.75rem;
             font-size: 0.95rem; resize: vertical; min-height: 80px; }
  textarea:focus { outline: none; border-color: #7c3aed; }
  .btn { background: #7c3aed; color: white; border: none; border-radius: 8px;
         padding: 0.6rem 1.5rem; font-size: 0.9rem; cursor: pointer;
         margin-top: 0.75rem; transition: background 0.2s; }
  .btn:hover { background: #6d28d9; }
  .btn:disabled { background: #374151; cursor: not-allowed; }
  .status-badge { display: inline-block; padding: 0.2rem 0.6rem;
                  border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
  .status-success { background: #064e3b; color: #34d399; }
  .status-partial { background: #78350f; color: #fbbf24; }
  .status-refused { background: #7f1d1d; color: #f87171; }
  .answer-box { background: #0f1117; border-left: 3px solid #7c3aed;
                padding: 1rem; border-radius: 0 8px 8px 0;
                font-size: 0.9rem; line-height: 1.7; white-space: pre-wrap; }
  .sentence-item { border: 1px solid #2d3748; border-radius: 8px;
                   padding: 0.75rem; margin-bottom: 0.5rem; }
  .label-support { border-left: 3px solid #34d399; }
  .label-refute { border-left: 3px solid #f87171; }
  .label-insufficient { border-left: 3px solid #fbbf24; }
  .source-tag { display: inline-block; background: #1e3a5f; color: #60a5fa;
                font-size: 0.7rem; padding: 0.1rem 0.5rem; border-radius: 4px;
                margin-right: 0.25rem; }
  .meta { font-size: 0.75rem; color: #64748b; margin-top: 0.5rem; }
  .refuse-box { background: #450a0a; border: 1px solid #7f1d1d;
                border-radius: 8px; padding: 1rem; color: #fca5a5; }
  .loading { color: #7c3aed; font-size: 0.9rem; }
</style>
</head>
<body>
<div class="container">
  <h1>🔍 私域可信问答智能体</h1>
  <p class="subtitle">细粒度证据核验 · 成因定向修复 · 可追溯输出</p>

  <div class="card">
    <h3 style="margin-bottom:0.75rem;font-size:0.95rem;color:#94a3b8">提问</h3>
    <textarea id="question" placeholder="请输入问题，例如：三号发动机的额定推力是多少？"></textarea>
    <button class="btn" onclick="doQuery()" id="qbtn">发送问题</button>
  </div>

  <div id="result" style="display:none">
    <div class="card" id="summary-card">
      <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem">
        <span id="status-badge" class="status-badge"></span>
        <span id="meta-text" class="meta"></span>
      </div>
      <div id="answer-text" class="answer-box"></div>
    </div>
    <div class="card" id="refuse-card" style="display:none">
      <div class="refuse-box" id="refuse-text"></div>
    </div>
    <div class="card">
      <h3 style="margin-bottom:0.75rem;font-size:0.95rem;color:#94a3b8">逐句核验详情</h3>
      <div id="sentences"></div>
    </div>
  </div>
</div>

<script>
async function doQuery() {
  const q = document.getElementById('question').value.trim();
  if (!q) return;
  const btn = document.getElementById('qbtn');
  btn.disabled = true; btn.textContent = '核验中…';
  document.getElementById('result').style.display = 'none';

  try {
    const resp = await fetch('/query', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: q})
    });
    const data = await resp.json();
    renderResult(data);
  } catch(e) {
    alert('请求失败: ' + e.message);
  } finally {
    btn.disabled = false; btn.textContent = '发送问题';
  }
}

function renderResult(data) {
  document.getElementById('result').style.display = '';
  const statusMap = {success:'✅ 全部验证', partial:'⚠️ 部分验证', refused:'❌ 拒答', error:'💥 错误'};
  const classMap = {success:'status-success', partial:'status-partial', refused:'status-refused', error:'status-refused'};
  const badge = document.getElementById('status-badge');
  badge.textContent = statusMap[data.status] || data.status;
  badge.className = 'status-badge ' + (classMap[data.status] || '');

  document.getElementById('meta-text').textContent =
    `支持率 ${(data.supported_ratio*100).toFixed(0)}%  ·  迭代 ${data.iterations} 次  ·  耗时 ${data.time_ms}ms`;

  const answerBox = document.getElementById('answer-text');
  const refuseCard = document.getElementById('refuse-card');
  if (data.status === 'refused') {
    answerBox.textContent = '（无输出，触发拒答）';
    refuseCard.style.display = '';
    document.getElementById('refuse-text').textContent = '拒答原因：' + (data.refuse_reason || '');
  } else {
    answerBox.textContent = data.answer || '';
    refuseCard.style.display = 'none';
  }

  const container = document.getElementById('sentences');
  container.innerHTML = '';
  (data.sentence_details || []).forEach(sv => {
    const div = document.createElement('div');
    div.className = 'sentence-item label-' + sv.label;
    const labelEmoji = {support:'✅', refute:'❌', insufficient:'⚠️'}[sv.label] || '?';
    const causeStr = sv.cause ? `<span style="color:#94a3b8;font-size:0.75rem"> · 成因: ${sv.cause}</span>` : '';
    const srcHtml = (sv.sources||[]).map(s =>
      `<span class="source-tag" title="${s.excerpt}">${s.source.split('/').pop()}${s.page?'p'+s.page:''}</span>`
    ).join('');
    div.innerHTML = `
      <div>${labelEmoji} <strong>${sv.sentence}</strong>${causeStr}</div>
      <div class="meta" style="margin-top:0.4rem">${srcHtml}${sv.rule_check !== 'pass' ? '规则: '+sv.rule_check : ''}</div>`;
    container.appendChild(div);
  });
}

document.getElementById('question').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) doQuery();
});
</script>
</body>
</html>"""


# ── 启动入口 ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    api_cfg = get_config().get("api", {})
    uvicorn.run(app, host=api_cfg.get("host", "0.0.0.0"), port=api_cfg.get("port", 8000))
