"""Minimal OpenAI-compatible chat client (SGLang, vLLM, Ollama) over plain HTTP.

No streaming. Assistant reasoning is sent back as `reasoning_content` so Qwen-style templates
can keep thinking in history. `FakeLLM` replays a scripted list of responses for tests.
"""
from __future__ import annotations

import base64
import json
import struct
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import requests

from arc3.types import LLMMessage

OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
RETRY_STATUS = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """Any failure to get a usable completion."""


class LLMTimeout(LLMError):
    """The request exceeded its timeout."""


@dataclass
class LLMResponse:
    content: str = ""
    reasoning: Optional[str] = None
    tool_calls: list[dict] = field(default_factory=list)   # OpenAI wire format
    usage: dict = field(default_factory=lambda: {
        "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0})
    finish_reason: str = "stop"


# ---------------------------------------------------------------- message helpers

def image_part(png: bytes) -> dict:
    """An OpenAI `image_url` content part carrying a PNG as a data URL."""
    return {"type": "image_url",
            "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def png_size(png: bytes) -> tuple[int, int]:
    """(width, height) from the PNG IHDR chunk."""
    if len(png) < 24 or png[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    return struct.unpack(">II", png[16:24])


def data_url_size(url: str) -> tuple[int, int]:
    """(width, height) of a base64 PNG data URL; decodes only the header."""
    b64 = url.split(",", 1)[1]
    return png_size(base64.b64decode(b64[:44]))


def tool_call(code: str, call_id: Optional[str] = None) -> dict:
    """A `python` tool call in OpenAI wire format."""
    return {"id": call_id or "call_" + uuid.uuid4().hex[:12], "type": "function",
            "function": {"name": "python", "arguments": json.dumps({"code": code})}}


def parse_tool_code(call: dict) -> tuple[Optional[str], Optional[str]]:
    """Return (code, error) for a `python` tool call. Lenient: raw non-JSON arguments are
    taken as the code itself, because small models sometimes skip the JSON wrapper."""
    fn = call.get("function") or {}
    if fn.get("name") not in (None, "python"):
        return None, f"unknown tool {fn.get('name')!r}; the only tool is `python`"
    args = fn.get("arguments")
    if isinstance(args, dict):
        code = args.get("code")
    else:
        try:
            parsed = json.loads(args or "")
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            code = parsed.get("code")
        elif args and args.strip() and not args.strip().startswith("{"):
            code = args
        else:
            return None, "tool arguments were not valid JSON with a `code` string"
    if not isinstance(code, str) or not code.strip():
        return None, "missing `code` argument"
    return code, None


def to_wire(msg: LLMMessage) -> dict:
    """LLMMessage -> OpenAI chat message dict."""
    out: dict[str, Any] = {"role": msg.role, "content": msg.content if msg.content is not None else ""}
    if msg.role == "assistant":
        if msg.reasoning:
            out["reasoning_content"] = msg.reasoning
        if msg.tool_calls:
            out["tool_calls"] = msg.tool_calls
    if msg.role == "tool":
        out["tool_call_id"] = msg.tool_call_id
        out["name"] = "python"
    return out


# ---------------------------------------------------------------- HTTP client

class LLMClient:
    """OpenAI-compatible /chat/completions client."""

    def __init__(self, base_url: str = "http://127.0.0.1:30000/v1", model: str = "default",
                 api_key: str = "EMPTY", enable_thinking: Optional[bool] = True,
                 max_retries: int = 4, backoff: float = 2.0, timeout: float = 900.0,
                 session: Optional[requests.Session] = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.enable_thinking = enable_thinking
        self.max_retries = max_retries
        self.backoff = backoff
        self.timeout = timeout
        self.http = session or requests.Session()

    def wait_ready(self, timeout: float = 1800.0, poll: float = 5.0) -> bool:
        """Block until the server answers /health (or /models). Returns False on timeout."""
        if getattr(self, "_ready", False):
            return True
        root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        end = time.time() + timeout
        while time.time() < end:
            for url in (root + "/health", self.base_url + "/models"):
                try:
                    if self.http.get(url, timeout=10).status_code == 200:
                        self._ready = True
                        return True
                except requests.RequestException:
                    pass
            time.sleep(poll)
        return False

    def chat(self, messages: list[Union[LLMMessage, dict]], tools: Optional[list[dict]] = None,
             max_tokens: int = 12_000, temperature: float = 0.7, top_p: float = 0.95,
             top_k: Optional[int] = 20, seed: Optional[int] = None,
             timeout: Optional[float] = None) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [to_wire(m) if isinstance(m, LLMMessage) else m for m in messages],
            "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if top_k is not None:
            body["top_k"] = top_k
        if seed is not None:
            body["seed"] = seed
        if self.enable_thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        data = self._post(body, timeout or self.timeout)
        return parse_response(data)

    def _post(self, body: dict, timeout: float) -> dict:
        url = self.base_url + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last: Exception = LLMError("no attempt made")
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(self.backoff * (2 ** (attempt - 1)))
            try:
                r = self.http.post(url, json=body, headers=headers, timeout=timeout)
            except requests.Timeout as e:
                # A timed-out generation is not retried here: the caller owns the time budget.
                raise LLMTimeout(f"request timed out after {timeout:.0f}s") from e
            except requests.ConnectionError as e:
                last = LLMError(f"connection error: {e}")
                continue
            if r.status_code in RETRY_STATUS:
                last = LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
                continue
            if r.status_code != 200:
                raise LLMError(f"HTTP {r.status_code}: {r.text[:500]}")
            try:
                return r.json()
            except ValueError as e:
                raise LLMError(f"invalid JSON response: {r.text[:300]}") from e
        raise last


def parse_response(data: dict) -> LLMResponse:
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"response has no choices: {str(data)[:300]}") from e
    msg = choice.get("message") or {}
    usage = data.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    calls = []
    for c in msg.get("tool_calls") or []:
        c = dict(c)
        c.setdefault("id", "call_" + uuid.uuid4().hex[:12])
        c["id"] = c["id"] or "call_" + uuid.uuid4().hex[:12]
        c.setdefault("type", "function")
        calls.append(c)
    return LLMResponse(
        content=msg.get("content") or "",
        reasoning=msg.get("reasoning_content") or msg.get("reasoning") or None,
        tool_calls=calls,
        usage={"prompt_tokens": int(usage.get("prompt_tokens") or 0),
               "completion_tokens": int(usage.get("completion_tokens") or 0),
               "cached_tokens": int(details.get("cached_tokens") or usage.get("cached_tokens") or 0)},
        finish_reason=choice.get("finish_reason") or "stop",
    )


def OllamaLLM(model: str = "qwen3.5:4b", **kw) -> LLMClient:
    """Same client pointed at a local Ollama server (protocol smoke tests only)."""
    kw.setdefault("enable_thinking", None)
    return LLMClient(base_url=OLLAMA_BASE_URL, model=model, **kw)


# ---------------------------------------------------------------- fake for tests

Script = Union[LLMResponse, Exception, Callable[[list], LLMResponse]]


class FakeLLM:
    """Replays a scripted sequence. Items may be LLMResponse, an Exception (raised), or a
    callable taking the wire messages and returning an LLMResponse. Records every request."""

    def __init__(self, script: list[Script], prompt_tokens: Optional[Callable[[list], int]] = None):
        self.script = list(script)
        self.calls: list[dict] = []
        self.prompt_tokens = prompt_tokens

    def chat(self, messages, tools=None, max_tokens=0, temperature=0.0, top_p=1.0,
             top_k=None, seed=None, timeout=None) -> LLMResponse:
        wire = [to_wire(m) if isinstance(m, LLMMessage) else m for m in messages]
        self.calls.append({"messages": json.loads(json.dumps(wire)), "tools": tools,
                           "max_tokens": max_tokens})
        if not self.script:
            raise LLMError("FakeLLM script exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        resp = item(wire) if callable(item) else item
        if self.prompt_tokens and not resp.usage.get("prompt_tokens"):
            resp.usage["prompt_tokens"] = self.prompt_tokens(wire)
        return resp

    @staticmethod
    def code(code: str, reasoning: str = "", tokens: int = 100) -> LLMResponse:
        return LLMResponse(content="", reasoning=reasoning or None, tool_calls=[tool_call(code)],
                           usage={"prompt_tokens": 0, "completion_tokens": tokens, "cached_tokens": 0},
                           finish_reason="tool_calls")

    @staticmethod
    def text(content: str, reasoning: str = "", tokens: int = 100) -> LLMResponse:
        return LLMResponse(content=content, reasoning=reasoning or None,
                           usage={"prompt_tokens": 0, "completion_tokens": tokens, "cached_tokens": 0})
