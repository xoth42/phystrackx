"""
The local model runner's client: any server with the OpenAI-compatible chat API and tool calls.
Standard library only. It streams, so a reply shows as it is written and Stop cuts it off.

- Ollama (the default): installed on this Mac. It runs models on the CPU through llama.cpp (the
  Radeon Pro 560X is not used). Measured here (i9-8950HK): qwen3:4b-instruct answers a tool
  call in 2-3 s (prompts ~650 tokens/s once warm, writing ~9 tokens/s). Thinking models are
  much slower here: qwen3:4b writes 500-950 tokens of reasoning before each call (60-125 s), and
  no setting turned that off. So an instruct model is the default.
- Strata (github.com/Niko1221/Strata): http://127.0.0.1:8080/v1, on a machine it supports:
  Apple Silicon, or a 12 GB+ NVIDIA/AMD GPU with 32 GB+ of RAM and VRAM together. Not this
  Intel Mac.
- llama.cpp's llama-server, LM Studio, vLLM: their /v1 URL.

PHYSTRACK_LLM_URL and PHYSTRACK_LLM_MODEL choose (the assistant panel also lists the runner's
models).
"""

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:11434/v1"   # Ollama
DEFAULT_MODEL = "qwen3:4b-instruct"


class LLMError(Exception):
    """The runner could not be reached, or refused the request (the message says which)."""


class Cancelled(Exception):
    """The user stopped the reply being written."""


class LLM:
    def __init__(self, url: str | None = None, model: str | None = None, key: str | None = None) -> None:
        self.url = (url or os.environ.get("PHYSTRACK_LLM_URL") or DEFAULT_URL).rstrip("/")
        self.model = model or os.environ.get("PHYSTRACK_LLM_MODEL") or DEFAULT_MODEL
        self.key = key or os.environ.get("PHYSTRACK_LLM_KEY") or "local"

    @property
    def is_ollama(self) -> bool:
        return ":11434" in self.url

    def _open(self, path: str, body: dict | None = None, timeout: float = 10.0):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.url + path, data, {
            "Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        return urllib.request.urlopen(request, timeout=timeout)

    def models(self) -> list[str]:
        """The runner's models (raises OSError if it cannot be reached)."""
        with self._open("/models", timeout=3.0) as response:
            return sorted(m["id"] for m in json.loads(response.read()).get("data", []))

    def reachable(self) -> bool:
        try:
            self.models()
            return True
        except (OSError, ValueError):
            return False

    def start_runner(self) -> bool:
        """Starts Ollama when it is the runner, installed and not running (the model stays loaded
        for 30 minutes after use). True if the runner can be reached."""
        if self.reachable():
            return True
        if not self.is_ollama or not shutil.which("ollama"):
            return False
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True, env={**os.environ, "OLLAMA_KEEP_ALIVE": "30m"})
        for _ in range(60):
            time.sleep(0.25)
            if self.reachable():
                return True
        return False

    def chat(self, messages: list, tools: list, on_text=None, cancelled=None,
             max_tokens: int = 400) -> dict:
        """
        One reply: {"content": text, "tool_calls": [{"id", "name", "arguments": dict, "bad": text}]}.
        `on_text(piece)` gets the text as it is written; `cancelled()` is asked between pieces
        (True: the reply stops, and Cancelled is raised). "bad" is set when a call's arguments
        are not JSON (they are then {}), so the caller can say so to the model.
        """
        body = {"model": self.model, "messages": messages, "tools": tools, "stream": True,
                "temperature": 0.2, "max_tokens": max_tokens}
        try:
            response = self._open("/chat/completions", body, timeout=600.0)
        except urllib.error.HTTPError as e:
            raise LLMError(f"the model runner refused the request ({e.code}): "
                           f"{e.read()[:300].decode('utf-8', 'replace')}") from None
        except OSError as e:
            raise LLMError(f"cannot reach the model runner at {self.url} ({e})") from None
        content, calls = [], {}
        with response:
            for raw in response:
                if cancelled is not None and cancelled():
                    raise Cancelled()
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        content.append(delta["content"])
                        if on_text is not None:
                            on_text(delta["content"])
                    # A call comes whole (Ollama) or in pieces (OpenAI, llama.cpp): its name once,
                    # its arguments as text in pieces
                    for tc in delta.get("tool_calls") or []:
                        slot = calls.setdefault(tc.get("index", len(calls)), {"id": "", "name": "", "args": ""})
                        slot["id"] = tc.get("id") or slot["id"]
                        fn = tc.get("function") or {}
                        slot["name"] = fn.get("name") or slot["name"]
                        args = fn.get("arguments")
                        if isinstance(args, dict):
                            slot["args"] = json.dumps(args)
                        elif args:
                            slot["args"] += args
        out = []
        for i, slot in sorted(calls.items()):
            entry = {"id": slot["id"] or f"call_{i}", "name": slot["name"], "arguments": {}}
            try:
                parsed = json.loads(slot["args"]) if slot["args"].strip() else {}
                entry["arguments"] = parsed if isinstance(parsed, dict) else {}
            except ValueError:
                entry["bad"] = slot["args"][:200]
            out.append(entry)
        return {"content": "".join(content), "tool_calls": out}
