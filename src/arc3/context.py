"""Chat history with a calibrated token estimate, block trimming and an image window.

Token estimate: text chars / ratio, where the ratio is calibrated online from the server's
reported prompt_tokens; images cost ceil(w/px)*ceil(h/px) + overhead tokens (Qwen3-VL merges
2x2 patches of 16 px, so one token per 32x32 px; a 640x640 board is ~400 tokens).

Trimming happens only when the budget is exceeded, then drains well below it; large, rare
evictions keep the server's prefix cache valid most of the time. It first strips reasoning from
assistant messages older than the last `keep_reasoning_turns` blocks (reasoning is ~75 % of the
generated tokens, and the code and tool results it led to stay), and only if that is not enough
drops whole blocks (a user turn plus the assistant/tool messages after it) from the oldest end.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Optional

from arc3.llm import data_url_size
from arc3.types import LLMMessage


@dataclass
class TokenModel:
    chars_per_token: float = 3.5        # initial guess; recalibrated from server usage
    image_px_per_token: int = 32
    image_overhead: int = 4             # vision start/end markers
    message_overhead: int = 5           # role tags per message
    min_ratio: float = 1.5
    max_ratio: float = 8.0
    smoothing: float = 0.3              # weight of each new observation


def _image_tokens(part: dict, tm: TokenModel) -> int:
    try:
        w, h = data_url_size(part["image_url"]["url"])
    except Exception:
        w, h = 640, 640
    return math.ceil(w / tm.image_px_per_token) * math.ceil(h / tm.image_px_per_token) + tm.image_overhead


def measure(msg: LLMMessage, tm: TokenModel) -> tuple[int, int]:
    """(text chars, image tokens) of a message."""
    chars, img = 0, 0
    if isinstance(msg.content, str):
        chars += len(msg.content)
    elif isinstance(msg.content, list):
        for part in msg.content:
            if part.get("type") == "image_url":
                img += _image_tokens(part, tm)
            else:
                chars += len(part.get("text", ""))
    if msg.reasoning:
        chars += len(msg.reasoning)
    if msg.tool_calls:
        chars += len(json.dumps(msg.tool_calls))
    return chars, img


def has_images(msg: LLMMessage) -> bool:
    return isinstance(msg.content, list) and any(p.get("type") == "image_url" for p in msg.content)


class ContextManager:
    def __init__(self, system_prompt: str, token_model: Optional[TokenModel] = None,
                 images_window: int = 2, keep_reasoning: bool = True, extra_chars: int = 0,
                 keep_reasoning_turns: int = 6):
        """`extra_chars`: text sent with every request outside the messages (tool schema).
        `keep_reasoning_turns`: blocks at the end whose reasoning a trim never strips."""
        self.tm = token_model or TokenModel()
        self.images_window = images_window
        self.keep_reasoning = keep_reasoning
        self.keep_reasoning_turns = keep_reasoning_turns
        self.extra_chars = extra_chars
        self.messages: list[LLMMessage] = []
        self.trims = 0
        # True after a trim dropped whole blocks. The agent prefixes its next opener with
        # prompts.trim_notice and clears it; add_user does not, so a nudge cannot swallow it.
        self.trimmed_since_last_user = False
        self.last_trim = {"stripped": 0, "dropped": 0}
        self._add(LLMMessage(role="system", content=system_prompt))

    # ------------------------------------------------------------ estimates

    @property
    def ratio(self) -> float:
        return self.tm.chars_per_token

    def estimate_tokens(self, msg: LLMMessage) -> int:
        chars, img = measure(msg, self.tm)
        return math.ceil(chars / self.tm.chars_per_token) + img + self.tm.message_overhead

    def total_tokens(self) -> int:
        self._refresh()
        return sum(m.tokens for m in self.messages) + math.ceil(self.extra_chars / self.tm.chars_per_token)

    def calibrate(self, prompt_tokens: int, messages: Optional[list[LLMMessage]] = None) -> None:
        """Update chars/token from the server's prompt_tokens for the request just sent."""
        msgs = self.messages if messages is None else messages
        if prompt_tokens <= 0 or not msgs:
            return
        chars, fixed = self.extra_chars, 0
        for m in msgs:
            c, img = measure(m, self.tm)
            chars += c
            fixed += img + self.tm.message_overhead
        text_tokens = prompt_tokens - fixed
        if text_tokens <= 0 or chars < 200:
            return
        observed = min(max(chars / text_tokens, self.tm.min_ratio), self.tm.max_ratio)
        a = self.tm.smoothing
        self.tm.chars_per_token = (1 - a) * self.tm.chars_per_token + a * observed
        self._refresh()

    def _refresh(self) -> None:
        for m in self.messages:
            m.tokens = self.estimate_tokens(m)

    # ------------------------------------------------------------ mutation

    def _add(self, msg: LLMMessage) -> LLMMessage:
        msg.tokens = self.estimate_tokens(msg)
        self.messages.append(msg)
        return msg

    def add_user(self, content) -> LLMMessage:
        msg = self._add(LLMMessage(role="user", content=content))
        self.apply_image_window()
        return msg

    def add_assistant(self, content: str, reasoning: Optional[str] = None,
                      tool_calls: Optional[list[dict]] = None) -> LLMMessage:
        return self._add(LLMMessage(role="assistant", content=content or "",
                                    reasoning=reasoning if self.keep_reasoning else None,
                                    tool_calls=tool_calls or None))

    def add_tool(self, tool_call_id: str, content: str) -> LLMMessage:
        return self._add(LLMMessage(role="tool", content=content, tool_call_id=tool_call_id))

    def apply_image_window(self) -> None:
        """Only the last `images_window` image-bearing user messages keep their images."""
        seen = 0
        for m in reversed(self.messages):
            if m.role != "user" or not has_images(m):
                continue
            seen += 1
            if seen > self.images_window:
                m.content = _strip_images(m.content)
                m.tokens = self.estimate_tokens(m)

    # ------------------------------------------------------------ trimming

    def blocks(self) -> list[tuple[int, int]]:
        """[start, end) index ranges of blocks after the system prompt. A block starts at a
        user message; anything before the first user message belongs to the first block."""
        starts = [i for i, m in enumerate(self.messages) if i > 0 and m.role == "user"]
        if not starts:
            return [(1, len(self.messages))] if len(self.messages) > 1 else []
        starts[0] = 1
        ends = starts[1:] + [len(self.messages)]
        return list(zip(starts, ends))

    def strip_old_reasoning(self) -> int:
        """Drop reasoning from assistant messages before the last `keep_reasoning_turns` blocks.
        Returns the number of messages changed."""
        blocks = self.blocks()
        if len(blocks) <= self.keep_reasoning_turns:
            return 0
        end = blocks[-self.keep_reasoning_turns][0] if self.keep_reasoning_turns else len(self.messages)
        changed = 0
        for m in self.messages[1:end]:
            if m.role == "assistant" and m.reasoning:
                m.reasoning = None
                m.tokens = self.estimate_tokens(m)
                changed += 1
        return changed

    def trim(self, budget_tokens: int, drain_tokens: int) -> int:
        """If over budget: strip old reasoning, then, if total is still above budget - drain, drop
        oldest blocks until it is not. The system prompt and the most recent block are always
        kept. Returns the number of messages changed (reasoning stripped + messages dropped), so
        any non-zero value means the prompt prefix changed; `last_trim` has the split."""
        self.last_trim = {"stripped": 0, "dropped": 0}
        total = self.total_tokens()
        if total <= budget_tokens:
            return 0
        target = max(budget_tokens - drain_tokens, 0)
        stripped = self.strip_old_reasoning()
        self.last_trim["stripped"] = stripped
        total = self.total_tokens()
        if total <= target:
            if stripped:
                self.trims += 1
            return stripped
        blocks = self.blocks()
        drop_until = 1
        for start, end in blocks[:-1]:
            if total <= target:
                break
            total -= sum(m.tokens for m in self.messages[start:end])
            drop_until = end
        dropped = drop_until - 1
        if dropped:
            del self.messages[1:drop_until]
            self.trimmed_since_last_user = True
        if dropped or stripped:
            self.trims += 1
        self.last_trim["dropped"] = dropped
        return stripped + dropped


def _strip_images(content: list) -> list:
    out = []
    for part in content:
        if part.get("type") == "image_url":
            out.append({"type": "text", "text": "[older image removed]"})
        else:
            out.append(part)
    return out
