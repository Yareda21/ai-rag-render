
# rag_api/main.py
import os
import re
import requests
import traceback
import json
import time
from typing import Tuple, List
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from dotenv import load_dotenv
from starlette.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
from supabase_rag import init_db_pool
from supabase_rag import retrieve_topk

load_dotenv()
EMBED_SERVICE_URL = os.getenv("EMBED_SERVICE_URL", "http://localhost:8000/embed")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free") 
OPENROUTER_BASE = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "2000")) 
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "800"))
SAFE_TOP_K = int(os.getenv("TOP_K", "4"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize database pool and other resources
    print("Starting up application...")
    await run_in_threadpool(init_db_pool)
    print("DB pool initialized.")
    
    # You can add other startup logic here
    # For example, checking dependencies or loading other models
    
    yield
    
    # Shutdown: Clean up resources
    print("Shutting down application...")
    # Add any cleanup code here if needed


class ChatRequest(BaseModel):
    prompt: str
    use_rag: bool = True

app = FastAPI(title="RAG API", lifespan=lifespan)


def is_embedding_service_ready(retries=5, delay=5):
    """Checks the health of the embedding service with retries."""
    try:
        # Note: We call the /ready endpoint, not /embed
        ready_url = EMBED_SERVICE_URL.replace("/embed", "/ready")
        for i in range(retries):
            response = requests.get(ready_url)
            if response.status_code == 200:
                print("Embedding service is ready.")
                return True
            print(f"Attempt {i+1} failed with status {response.status_code}. Retrying in {delay}s.")
            time.sleep(delay)
        print("Embedding service is not ready after all retries.")
        return False
    except requests.RequestException as e:
        print(f"Embedding service connection error: {e}. Cannot check readiness.")
        return False


def sanitize_text(s: str) -> str:
    """Remove control characters and collapse excessive whitespace."""
    if not isinstance(s, str):
        return ""
    s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u0080-\uFFFF]+", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def build_trimmed_context(retrieved: List[dict], max_total=MAX_CONTEXT_CHARS, per_chunk=MAX_CHUNK_CHARS):
    """Sanitize and trim retrieved {"text":...} into a single context string."""
    parts = []
    total = 0
    for r in retrieved:
        txt = sanitize_text(r.get("text", ""))
        if not txt:
            continue
        if len(txt) > per_chunk:
            txt = txt[:per_chunk] + " …(truncated)"
        if total + len(txt) > max_total:
            remaining = max(0, max_total - total)
            if remaining <= 10:
                break
            parts.append(txt[:remaining] + " …(truncated)")
            total += remaining
            break
        parts.append(txt)
        total += len(txt)
    return "\n\n".join(parts)


def get_query_embedding_via_embed_service(text: str) -> List[float]:
    """Calls the embedding service to get an embedding for the given text."""
    try:
        print(f"Attempting to call embedding service at: {EMBED_SERVICE_URL}")
        resp = requests.post(EMBED_SERVICE_URL, json={"texts":[text]}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        emb = data.get("embeddings", [])
        if not emb:
            raise ValueError("Embedding service returned an empty embedding list.")
        print("Successfully received embedding from service.")
        return emb[0]
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error from embedding service: {errh}")
        raise RuntimeError(f"HTTP Error from embedding service: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Connection Error from embedding service: {errc}")
        raise RuntimeError(f"Connection Error from embedding service: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error from embedding service: {errt}")
        raise RuntimeError(f"Timeout Error from embedding service: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"Unknown requests error from embedding service: {err}")
        raise RuntimeError(f"Unknown requests error from embedding service: {err}")
    except (json.JSONDecodeError, ValueError) as err:
        print(f"JSON or value error from embedding service response: {err}")
        raise RuntimeError(f"Invalid response from embedding service: {err}")


def _post_and_debug(url: str, headers: dict, payload: dict, timeout: int = 60) -> Tuple[requests.Response, str]:
    """Post to OpenRouter and return (response, raw_text)."""
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error calling OpenRouter: {e}") from e
    text = r.text or ""
    return r, text


def call_openrouter_llm(prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
    """Calls the OpenRouter LLM API with the given prompt."""
    if not OPENROUTER_KEY:
        raise RuntimeError("OPENROUTER_KEY not set")

    url = f"{OPENROUTER_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful tutor."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    r, raw_text = _post_and_debug(url, headers, payload)
    
    if not r.ok:
        print(f"[OpenRouter debug] status: {r.status_code}")
        print(f"[OpenRouter debug] response body: {raw_text[:2000]}")
        raise RuntimeError(f"OpenRouter returned {r.status_code}. Body: {raw_text[:2000]}")
        
    try:
        data = r.json()
    except Exception:
        return raw_text

    choices = data.get("choices", [])
    if choices and isinstance(choices, list) and len(choices) > 0:
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message") or {}
            if isinstance(msg, dict) and msg.get("content"):
                return msg["content"]
            if first.get("text"):
                return first.get("text")
    for k in ("response", "text", "content"):
        if isinstance(data.get(k), str):
            return data.get(k)
    return json.dumps(data)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ai/chat")
async def ai_chat(req: ChatRequest):
    try:
        if req.use_rag:
            if not await run_in_threadpool(is_embedding_service_ready):
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Embedding service is not ready. Try again in a moment.")
            
            query_emb = await run_in_threadpool(get_query_embedding_via_embed_service, req.prompt)
            print(f"Query embedding length: {len(query_emb)}") 
            
            retrieved = await run_in_threadpool(retrieve_topk, query_emb, SAFE_TOP_K)
            print("Retrieved chunks:", retrieved)
            
            context = build_trimmed_context(retrieved, max_total=MAX_CONTEXT_CHARS, per_chunk=MAX_CHUNK_CHARS)
            
            if not context:
                prompt = req.prompt
                full_prompt_is_rag = False
            else:
                prompt = (
                    "You are a helpful tutor. Use ONLY the context below to answer the question. If not present, say \"I don't know.\"\n\n"
                    f"CONTEXT:\n{context}\n\nQuestion: {req.prompt}\n\nAnswer:"
                )
                full_prompt_is_rag = True

            answer = await run_in_threadpool(call_openrouter_llm, prompt)
            return {"answer": answer}
        else:
            answer = await run_in_threadpool(call_openrouter_llm, req.prompt)
            return {"answer": answer}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

