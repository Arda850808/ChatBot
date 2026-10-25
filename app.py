import asyncio
import io
import json
import os
from datetime import date
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

app = FastAPI(title="ChatBot")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

LEVELS = ("off", "low", "medium", "high")
MAX_SOURCES = 4
PAGE_CHARS = 4000
MAX_DOC_CHARS = 60000
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGES = 4
MAX_IMAGE_B64 = 6_000_000
ALLOWED_IMAGE_MIME = {"image/jpeg", "image/png", "image/webp", "image/gif"}

# Tek kaynak: model listesi, görsel desteği ve .env anahtar adları burada
PROVIDERS = {
    "groq": {
        "name": "Groq API",
        "env": "GROQ_API_KEY",
        "models": [
            {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B Versatile", "vision": False},
            {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B Instant", "vision": False},
            {"id": "meta-llama/llama-4-scout-17b-16e-instruct", "name": "Llama 4 Scout 17B", "vision": True},
            {"id": "openai/gpt-oss-120b", "name": "GPT-OSS 120B", "vision": False},
            {"id": "openai/gpt-oss-20b", "name": "GPT-OSS 20B", "vision": False},
            {"id": "qwen/qwen3-32b", "name": "Qwen 3 32B", "vision": False},
        ],
    },
    "gemini": {
        "name": "Gemini API",
        "env": "GEMINI_API_KEY",
        "models": [
            {"id": "gemini-3.8-flash", "name": "Gemini 3.8 Flash", "vision": True},
            {"id": "gemini-3.7-flash", "name": "Gemini 3.7 Flash", "vision": True},
            {"id": "gemini-3.6-flash", "name": "Gemini 3.6 Flash", "vision": True},
            {"id": "gemini-3.5-flash", "name": "Gemini 3.5 Flash", "vision": True},
            {"id": "gemini-3.5-flash-lite", "name": "Gemini 3.5 Flash-Lite", "vision": True},
            {"id": "gemini-3.1-flash-lite", "name": "Gemini 3.1 Flash-Lite", "vision": True},
            {"id": "gemini-3-flash-preview", "name": "Gemini 3 Flash (Preview)", "vision": True},
            {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro", "vision": True},
            {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash", "vision": True},
            {"id": "gemini-2.5-flash-lite", "name": "Gemini 2.5 Flash-Lite", "vision": True},
        ],
    },
    "openrouter": {
        "name": "OpenRouter API",
        "env": "OPENROUTER_API_KEY",
        "models": [
            {"id": "deepseek/deepseek-r1:free", "name": "DeepSeek R1 (Free)", "vision": False},
            {"id": "qwen/qwen-2.5-coder-32b-instruct:free", "name": "Qwen 2.5 Coder 32B (Free)", "vision": False},
        ],
    },
}

# Gemini 3 modellerinde "minimal" düşünme seviyesi yalnızca bu modellerde destekleniyor
GEMINI_MINIMAL_OK = {
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
}
OPENROUTER_NATIVE = {"deepseek/deepseek-r1:free"}
GROQ_EFFORT = {"off": "low", "low": "low", "medium": "medium", "high": "high"}

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
THINK_INSTRUCTIONS = {
    "low": (
        "Yanıtından önce kısa bir düşünce süreci yürüt. Bu düşünceyi yalnızca "
        "<think> ve </think> etiketleri arasına yaz. Etiketlerin dışında düşünce, plan "
        "veya iç konuşma paylaşma. Etiketi kapattıktan sonra yalnızca nihai yanıtı ver."
    ),
    "medium": (
        "Yanıtından önce adım adım ve dikkatli düşün. Düşünce sürecini yalnızca "
        "<think> ve </think> etiketleri arasına yaz. Etiketlerin dışında düşünce, plan "
        "veya iç konuşma paylaşma. Etiketi kapattıktan sonra yalnızca nihai yanıtı ver."
    ),
    "high": (
        "Yanıtından önce sorunu parçalara ayırarak derinlemesine düşün, her adımı kontrol "
        "et ve çelişkileri gider. Düşünce sürecini yalnızca <think> ve </think> etiketleri "
        "arasına yaz. Etiketlerin dışında düşünce, plan veya iç konuşma paylaşma. Etiketi "
        "kapattıktan sonra yalnızca nihai yanıtı ver."
    ),
}


class ProviderError(Exception):
    pass


class ChatMessage(BaseModel):
    role: str
    content: str = ""
    images: List[str] = []


class ChatRequest(BaseModel):
    provider: str
    model: str
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 4096
    system_prompt: str = ""
    messages: List[ChatMessage]
    reasoning: str = "off"
    web_search: bool = False


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def get_api_key(provider: str) -> str:
    env_name = PROVIDERS[provider]["env"]
    key = os.getenv(env_name, "").strip()
    if not key:
        raise ProviderError(f"{env_name} .env dosyasında tanımlı değil.")
    return key


def model_info(provider: str, model_id: str) -> Optional[dict]:
    for m in PROVIDERS.get(provider, {}).get("models", []):
        if m["id"] == model_id:
            return m
    return None


def image_supported(provider: str, model_id: str) -> bool:
    info = model_info(provider, model_id)
    if info is None:  # Özel model ID: kullanıcı görsel desteğini ayarlardan onaylar
        return True
    return bool(info.get("vision"))


def native_reasoning_kind(provider: str, model: str) -> Optional[str]:
    if provider == "gemini":
        if model.startswith("gemini-3"):
            return "gemini3"
        if model.startswith("gemini-2.5"):
            return "gemini25"
    elif provider == "groq":
        if model.startswith("openai/gpt-oss"):
            return "groq_effort"
        if model.startswith("qwen/qwen3"):
            return "groq_qwen"
    elif provider == "openrouter" and model in OPENROUTER_NATIVE:
        return "openrouter"
    return None


def gemini_thinking_config(kind: str, model: str, level: str) -> dict:
    include = level != "off"
    if kind == "gemini3":
        if level == "off":
            tl = "minimal" if model in GEMINI_MINIMAL_OK else "low"
        else:
            tl = level
        return {"thinkingLevel": tl, "includeThoughts": include}
    if level == "off":
        budget = 128 if "pro" in model else 0  # 2.5 Pro düşünmeyi tamamen kapatamaz
    else:
        budget = {"low": 1024, "medium": 8192, "high": 24576}[level]
    return {"thinkingBudget": budget, "includeThoughts": include}


def _partial_suffix(s: str, tag: str) -> int:
    for k in range(min(len(tag) - 1, len(s)), 0, -1):
        if s.endswith(tag[:k]):
            return k
    return 0


class ThinkSplitter:
    """<think>...</think> bloklarını düşünce kanalına yönlendirir.
    Etiketler parçalara bölünse bile doğru çalışır."""

    def __init__(self):
        self.buf = ""
        self.in_think = False

    def feed(self, text: str) -> List[Tuple[str, str]]:
        self.buf += text
        out: List[Tuple[str, str]] = []
        while True:
            if not self.in_think:
                idx = self.buf.find(THINK_OPEN)
                if idx == -1:
                    cut = len(self.buf) - _partial_suffix(self.buf, THINK_OPEN)
                    if cut > 0:
                        out.append(("content", self.buf[:cut]))
                    self.buf = self.buf[cut:]
                    break
                if idx > 0:
                    out.append(("content", self.buf[:idx]))
                self.buf = self.buf[idx + len(THINK_OPEN):]
                self.in_think = True
            else:
                idx = self.buf.find(THINK_CLOSE)
                if idx == -1:
                    cut = len(self.buf) - _partial_suffix(self.buf, THINK_CLOSE)
                    if cut > 0:
                        out.append(("reasoning", self.buf[:cut]))
                    self.buf = self.buf[cut:]
                    break
                if idx > 0:
                    out.append(("reasoning", self.buf[:idx]))
                self.buf = self.buf[idx + len(THINK_CLOSE):]
                self.in_think = False
        return out

    def flush(self) -> List[Tuple[str, str]]:
        kind = "reasoning" if self.in_think else "content"
        rest, self.buf = self.buf, ""
        return [(kind, rest)] if rest else []


def last_user_text(msgs: List[ChatMessage]) -> str:
    for m in reversed(msgs):
        if m.role == "user":
            return (m.content or "")[:300]
    return ""


async def search_web(query: str) -> List[dict]:
    if DDGS is None:
        raise ProviderError("ddgs kurulu değil. Çalıştır: pip install ddgs")

    def _run():
        return DDGS(timeout=10).text(query, max_results=MAX_SOURCES)

    return await asyncio.wait_for(asyncio.to_thread(_run), timeout=25)


async def fetch_page_text(url: str) -> str:
    if DDGS is None:
        return ""

    def _run():
        data = DDGS(timeout=10).extract(url)
        if isinstance(data, dict):
            return str(data.get("content") or "")
        return str(data or "")

    try:
        text = await asyncio.wait_for(asyncio.to_thread(_run), timeout=20)
    except Exception:
        return ""
    return text.strip()


def build_search_context(sources: List[dict]) -> str:
    parts = [
        "Aşağıda bu soru için yapılan web araştırmasının sonuçları var. Bilgilerini öncelikle "
        "bu kaynaklara dayandır ve kullandığın bilginin kaynağını URL ile belirt. Kaynaklarda "
        "yer almayan bilgiyi kesin bilgi gibi sunma."
    ]
    for i, s in enumerate(sources, 1):
        parts.append(f"[{i}] {s['title']}\nURL: {s['url']}\nİçerik:\n{s['text']}")
    return "\n\n".join(parts)


ALLOWED_IMAGE_MIME_SET = ALLOWED_IMAGE_MIME


def parse_data_url(url: str) -> Tuple[str, str]:
    if not url.startswith("data:") or ";base64," not in url:
        raise ProviderError("Geçersiz görsel verisi.")
    header, b64 = url.split(",", 1)
    mime = header[5:].split(";")[0]
    if mime not in ALLOWED_IMAGE_MIME_SET:
        raise ProviderError(f"Desteklenmeyen görsel türü: {mime}")
    if len(b64) > MAX_IMAGE_B64:
        raise ProviderError("Görsel çok büyük (en fazla yaklaşık 4 MB).")
    return mime, b64


def openai_message(m: ChatMessage, allow_images: bool) -> dict:
    if not (allow_images and m.images):
        return {"role": m.role, "content": m.content}
    content = [{"type": "text", "text": m.content.strip() or "Bu görseli incele."}]
    for img in m.images:
        parse_data_url(img)
        content.append({"type": "image_url", "image_url": {"url": img}})
    return {"role": m.role, "content": content}


def gemini_parts(m: ChatMessage, allow_images: bool) -> list:
    text = m.content.strip() or ("Bu görseli incele." if allow_images and m.images else " ")
    parts: list = [{"text": text}]
    if allow_images:
        for img in m.images:
            mime, b64 = parse_data_url(img)
            parts.append({"inline_data": {"mime_type": mime, "data": b64}})
    return parts


async def stream_openai_compatible(client, url, headers, payload) -> AsyncIterator[Tuple[str, str]]:
    async with client.stream("POST", url, headers=headers, json=payload) as resp:
        if resp.status_code != 200:
            body = (await resp.aread()).decode("utf-8", errors="replace")
            raise ProviderError(f"HTTP {resp.status_code}: {body[:800]}")
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if obj.get("error"):
                raise ProviderError(str(obj["error"])[:800])
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            reasoning = delta.get("reasoning") or delta.get("reasoning_content")
            if reasoning:
                yield ("reasoning", reasoning)
            content = delta.get("content")
            if content:
                yield ("content", content)


async def stream_gemini(client, url, headers, payload) -> AsyncIterator[Tuple[str, str]]:
    async with client.stream("POST", url, headers=headers, json=payload) as resp:
        if resp.status_code != 200:
            body = (await resp.aread()).decode("utf-8", errors="replace")
            raise ProviderError(f"HTTP {resp.status_code}: {body[:800]}")
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data:
                continue
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if obj.get("error"):
                raise ProviderError(str(obj["error"])[:800])
            for cand in obj.get("candidates") or []:
                parts = (cand.get("content") or {}).get("parts") or []
                for part in parts:
                    text = part.get("text")
                    if not text:
                        continue
                    yield ("reasoning" if part.get("thought") else "content", text)


async def stream_model(client, req: ChatRequest, api_key: str, level: str, kind: Optional[str],
                       system_text: str, msgs: List[ChatMessage], image_index: int):
    thinking_on = level != "off"
    max_out = req.max_tokens
    if thinking_on:
        # Düşünme tokenları çıktı bütçesinden yer kaplar; bütçeyi yükselt
        max_out = max(req.max_tokens, 16384 if req.provider == "gemini" else 8192)

    if req.provider == "gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{req.model}:streamGenerateContent?alt=sse"
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        contents = [
            {"role": "model" if m.role == "assistant" else "user",
             "parts": gemini_parts(m, i == image_index)}
            for i, m in enumerate(msgs)
        ]
        gen = {"temperature": req.temperature, "topP": req.top_p, "maxOutputTokens": max_out}
        if kind in ("gemini3", "gemini25"):
            gen["thinkingConfig"] = gemini_thinking_config(kind, req.model, level)
        payload = {"contents": contents, "generationConfig": gen}
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}
        async for item in stream_gemini(client, url, headers, payload):
            yield item
        return

    if req.provider == "groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
    else:
        url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    wire = [openai_message(m, i == image_index) for i, m in enumerate(msgs)]
    out_msgs = ([{"role": "system", "content": system_text}] if system_text else []) + wire
    payload = {
        "model": req.model,
        "messages": out_msgs,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "stream": True,
    }
    if req.provider == "groq":
        payload["max_completion_tokens"] = max_out
        if kind == "groq_effort":
            payload["reasoning_effort"] = GROQ_EFFORT[level]
            payload["reasoning_format"] = "parsed"
        elif kind == "groq_qwen":
            payload["reasoning_effort"] = "none" if level == "off" else "default"
            payload["reasoning_format"] = "parsed"
    else:
        payload["max_tokens"] = max_out
        if kind == "openrouter":
            payload["reasoning"] = {"exclude": True} if level == "off" else {"effort": level}
    async for item in stream_openai_compatible(client, url, headers, payload):
        yield item


async def generate_events(req: ChatRequest):
    try:
        if req.provider not in PROVIDERS:
            raise ProviderError("Bilinmeyen sağlayıcı.")
        api_key = get_api_key(req.provider)

        msgs = [m for m in req.messages if m.role in ("user", "assistant")]
        if not msgs:
            raise ProviderError("Gönderilecek mesaj yok.")

        last_user = max((i for i, m in enumerate(msgs) if m.role == "user"), default=-1)
        image_index = -1
        if last_user >= 0 and msgs[last_user].images:
            if len(msgs[last_user].images) > MAX_IMAGES:
                raise ProviderError(f"Bir mesajda en fazla {MAX_IMAGES} görsel gönderilebilir.")
            if not image_supported(req.provider, req.model):
                raise ProviderError("Seçili model görsel analizini desteklemiyor.")
            image_index = last_user

        level = req.reasoning if req.reasoning in LEVELS else "off"
        thinking_on = level != "off"
        kind = native_reasoning_kind(req.provider, req.model)
        prompt_think = thinking_on and kind is None

        system_parts = [f"Bugünün tarihi: {date.today().isoformat()}."]
        if req.system_prompt.strip():
            system_parts.append(req.system_prompt.strip())

        if req.web_search:
            query = last_user_text(msgs)
            yield sse({"type": "search", "phase": "start", "query": query})
            sources: List[dict] = []
            if not query:
                yield sse({"type": "search", "phase": "error", "message": "Arama için metin gerekli"})
            else:
                try:
                    results = await search_web(query)
                except Exception as e:
                    results = []
                    yield sse({"type": "search", "phase": "error", "message": str(e) or type(e).__name__})
                for r in results[:MAX_SOURCES]:
                    url = r.get("href") or r.get("url")
                    if not url:
                        continue
                    yield sse({"type": "search", "phase": "visit", "url": url})
                    text = await fetch_page_text(url) or str(r.get("body", ""))
                    sources.append({"title": r.get("title") or url, "url": url, "text": text[:PAGE_CHARS]})
            yield sse({"type": "search", "phase": "done", "count": len(sources)})
            if sources:
                system_parts.append(build_search_context(sources))

        if prompt_think:
            system_parts.append(THINK_INSTRUCTIONS[level])
        system_text = "\n\n".join(system_parts)

        splitter = ThinkSplitter()
        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=15.0)) as client:
            async for out_kind, text in stream_model(client, req, api_key, level, kind,
                                                     system_text, msgs, image_index):
                if out_kind == "reasoning":
                    if thinking_on:
                        yield sse({"type": "reasoning", "delta": text})
                    continue
                for k, t in splitter.feed(text):
                    if k == "reasoning":
                        if thinking_on:
                            yield sse({"type": "reasoning", "delta": t})
                    else:
                        yield sse({"type": "content", "delta": t})
            for k, t in splitter.flush():
                if k == "reasoning":
                    if thinking_on:
                        yield sse({"type": "reasoning", "delta": t})
                else:
                    yield sse({"type": "content", "delta": t})
        yield sse({"type": "done"})
    except ProviderError as e:
        yield sse({"type": "error", "message": str(e)})
    except httpx.HTTPError as e:
        yield sse({"type": "error", "message": f"Ağ hatası: {e}"})
    except Exception as e:
        yield sse({"type": "error", "message": f"Sunucu hatası: {e}"})


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/config")
async def config():
    # Anahtarların kendisi asla gönderilmez, yalnızca tanımlı olup olmadığı
    return {
        "providers": {
            pid: {
                "name": p["name"],
                "configured": bool(os.getenv(p["env"], "").strip()),
                "models": p["models"],
            }
            for pid, p in PROVIDERS.items()
        }
    }


@app.post("/api/extract")
async def extract_pdf(file: UploadFile = File(...)):
    data = await file.read()
    name = file.filename or "dosya.pdf"
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Dosya 15 MB sınırını aşıyor.")
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Bu uç yalnızca PDF dosyaları içindir.")
    if PdfReader is None:
        raise HTTPException(status_code=500, detail="pypdf kurulu değil. Çalıştır: pip install pypdf")
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [(p.extract_text() or "").strip() for p in reader.pages]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF okunamadı: {e}")
    text = "\n\n".join(t for t in pages if t)
    return {
        "name": name,
        "pages": len(pages),
        "text": text[:MAX_DOC_CHARS],
        "truncated": len(text) > MAX_DOC_CHARS,
    }


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    return StreamingResponse(
        generate_events(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )