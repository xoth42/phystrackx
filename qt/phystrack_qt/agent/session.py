"""
The in-app assistant: a local model (llm.py) using the agent tools, in each window's Assistant dock.

Its lifetime, kept simple so a small model cannot get it wrong:
- One conversation per app: every window's dock shows it, as the assistant moves between windows.
  The user sends a request; the session runs it on a worker thread until the model answers
  without a tool call (or after MAX_CALLS calls).
- Each turn sends the system prompt and the tool list, the same every turn (so the runner keeps
  them cached and only reads what is new), then the conversation, with old tool results cut.
- Tools run through bridge.invoke, as MCP calls do: the same checks, history marks and Space.
  The model gets a subset of them (MODEL_TOOLS: the fine-grained ones are left to bigger agents)
  and ask_user, which shows a question with buttons and waits for the answer.
- Stop (Space, or the button) cuts off the reply being written or the tool call under way, and
  the session waits. Start (with a note, if typed) calls resume and tells the model what the user
  changed meanwhile, then goes on. End drops the request.
- A thinking model's reasoning goes to the transcript (the docks show it when Reasoning is on),
  never back to the model.
- The model runner (runner.py) is made ready at launch (setup) and before each request. Ollama
  missing, or its model not downloaded: a question asks (y or n) before anything is installed or
  downloaded. Declined, the request says why.

QUICK-AND-DIRTY (see docs/AGENT_PLAN.md): one module-level session; context is trimmed by
message count, not tokens; the transcript is re-rendered whole on each change.
"""

import json
import threading

from PySide6.QtCore import QObject, Signal

from .. import settings as app_settings
from .. import windows
from . import bridge
from .llm import LLM, Cancelled, LLMError
from .runner import Runner
from .specs import SPECS, json_schema

MODEL_TOOLS = ["windows", "videos", "state", "recipe", "open", "run_recipe", "plots", "data", "revert"]
MAX_CALLS = 40          # Tool calls in one request before the session gives up
RESULT_CHARS = 1500     # A tool result, as the model gets it (the chat shows it whole)
OLD_RESULT_CHARS = 160  # A tool result KEEP_FULL messages ago, as the model gets it
KEEP_FULL = 12
MAX_MESSAGES = 80       # Older whole requests are dropped past this

SYSTEM = """You are the assistant in PhysTrackX, a video-tracking app for physics labs. Each trial video is in its own window (w1, w2...). The user did one trial by hand; you repeat that work on other trials with the tools, and answer questions about the windows.

To repeat the work of a trial on other trials:
1. Find the source window wX. If the request names another trial to copy (e.g. "like trial 1"), open that trial's project file first (videos lists it, marked project): that window is the source. Otherwise the source is the window that did the work.
2. recipe(src=wX): read its steps. They repeat everything the source did, so do them as they are. Ask the user only about a choice the request leaves open that changes the result (e.g. which objects: with ask_user, short, with options). Never ask what the source did, and never ask again what the user already answered.
3. For each trial: open(video) (a window already showing it is reused), then run_recipe(src=wX), with only=[numbers] if the user wants only some objects.
4. If run_recipe returns "check": fix it if you can (a fit: look at data(kind=r), then plots with range=[t0, t1] starting at the drop's low and ending before the motion is disturbed), else ask_user. Then run_recipe again with start=next.
5. End with a short summary: each trial's period T ± dT (s) and R².

Rules: act with tool calls; never describe a step you are about to take, make the call. Use only window ids and file names that tools returned (leave w out to use the current window). One tool call at a time. If a result has "stopped", make no more calls. Keep messages short."""

# A reply with no tool call ends the request. A small model sometimes writes what it will do next
# instead of doing it ("Next, I will run..."): such a reply gets a nudge to make the call, up to
# NUDGES times a request. QUICK-AND-DIRTY: recognised by these words, not by the model.
NUDGES = 2
ANNOUNCES = ("next,", "next i will", "i will now", "i'll now", "now i will", "now i'll", "let's begin",
             "let's start", "let me now", "i am going to", "i'm going to", "begin by", "start by")
OFFERS = ("let me know", "would you like", "do you want", "?")   # Questions to the user end the request
NUDGE = "Go on: make the next tool call now. Do not describe it."

ASK_USER = {"type": "function", "function": {
    "name": "ask_user",
    "description": "Ask the user a short question and wait for the answer. options: up to 4 short answers, shown as buttons.",
    "parameters": {"type": "object", "properties": {
        "question": {"type": "string"},
        "options": {"type": "array", "items": {"type": "string"}}}, "required": ["question"]}}}


def tool_defs() -> list:
    return [{"type": "function", "function": {"name": name, "description": SPECS[name][0],
                                              "parameters": json_schema(SPECS[name][1])}}
            for name in MODEL_TOOLS] + [ASK_USER]


def _payload(result: dict):
    return result["ok"] if "ok" in result else result


def _compact(result: dict, limit: int | None = RESULT_CHARS) -> str:
    """The result as JSON, cut at limit characters (None: whole)."""
    text = json.dumps(_payload(result), separators=(",", ":"), ensure_ascii=False, default=str)
    return text if limit is None or len(text) <= limit else text[:limit] + "…"


def _summary(name: str, result: dict) -> str:
    """For the chat, whole: what a tool call came to."""
    if "error" in result:
        return f"✗ {result['error']}"
    if "stopped" in result:
        return "■ stopped"
    out = _payload(result)
    if name == "run_recipe" and isinstance(out, dict):
        done = ", ".join(step[1] for step in out.get("done", []))
        if "check" in out:
            return f"done: {done or '-'}; check at step {out['check'][0]}: {out['check'][2]}"
        if "failed" in out:
            return f"done: {done or '-'}; failed at step {out['failed'][0]}: {out['failed'][2]}"
        if "stopped" in out:
            return f"done: {done or '-'}; stopped"
        return f"done: {done}"
    return _compact(result, None)


class _Ended(Exception):
    """The user ended the request while it was stopped."""


class AgentSession(QObject):
    # (kind, text): user | assistant | part | reasoning | thought_part | tool | result | ask | status | error | clear
    event = Signal(str, str)
    state_changed = Signal(str)    # idle | setup | thinking | acting | waiting | stopped
    view_changed = Signal()        # Reasoning shown or hidden, in every dock

    def __init__(self, llm=None) -> None:
        super().__init__()
        bridge.ensure_invoker()
        self.llm = llm or LLM()
        self.runner = Runner(self.llm)
        self.tools = tool_defs()
        self.messages: list = []         # The conversation, without the system prompt
        self.transcript: list = []       # (kind, text) as the docks show it
        self.state = "idle"
        self._answer: str | None = None
        self._answered = threading.Event()
        self._go = threading.Event()
        self._note = ""
        self._end = False
        self._thread: threading.Thread | None = None
        self.settings = app_settings.shared()   # Show reasoning, and the reply length cap
        self.settings.changed.connect(self._setting_changed)

    # --- From the docks (Qt thread) ----------------------------------------------------------------
    def send(self, text: str) -> None:
        """The user's message: a new request when idle, the answer to a question, or (stopped)
        the note that goes with Start."""
        text = (text or "").strip()
        if self.state == "waiting":
            if text:
                self._emit("user", text)
                self._answer = text
                self._set_state("thinking")   # At once: a second click or Enter answers nothing
                self._answered.set()
            return
        if self.state == "setup":
            if text:
                self._emit("status", "The model runner is still being set up. "
                                     "Send your message when it is ready.")
            return
        if self.state == "stopped":
            if text:
                self._emit("user", text)
            self._note = text
            self._go.set()
            return
        if self.state != "idle" or not text:
            return
        if windows.AGENT.stopped:   # A stop left from before (an MCP agent's): a new request starts afresh
            windows.AGENT.resume()
        self.messages.append({"role": "user", "content": text})
        self._emit("user", text)
        self._set_state("thinking")
        self._thread = threading.Thread(target=self._run, name="assistant", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """The Stop button: as Space does."""
        if self.state in ("thinking", "acting", "waiting"):
            windows.AGENT.stop()

    def end(self) -> None:
        """Drops the request (while stopped)."""
        if self.state == "stopped":
            self._end = True
            self._go.set()

    @property
    def show_reasoning(self) -> bool:
        """The docks show the model's reasoning (a setting, kept between runs)."""
        return self.settings.get("show_reasoning")

    def set_show_reasoning(self, show: bool) -> None:
        self.settings.set("show_reasoning", bool(show))   # view_changed follows

    def _setting_changed(self, key: str) -> None:
        if key == "show_reasoning":
            self.view_changed.emit()

    def clear(self) -> None:
        if self.state == "idle":
            self.messages.clear()
            self.transcript.clear()
            self.event.emit("clear", "")

    def warm_up(self) -> None:
        """At launch: brings the model runner up in the background, so the first request does not
        wait for it. An install or a download is asked for first, in the dock."""
        self.runner.start()
        if self.state == "idle":
            self._set_state("setup")
            threading.Thread(target=self._setup, name="assistant-setup", daemon=True).start()

    def shutdown(self) -> None:
        """At quit: stops the server the app started (not one it found running)."""
        self.runner.stop()

    # --- The worker thread -------------------------------------------------------------------------
    def _setup(self) -> None:
        try:
            self._ready_runner(report=False)
        finally:
            self._busy(False)
            self._set_state("idle")

    def _ready_runner(self, report: bool) -> bool:
        """Makes the model runner ready, asking (y or n) before an install or a download. When it
        cannot, the reason goes to the transcript if report is set; the dock's label shows it either way."""
        for _ in range(4):   # Install, start, download: three rounds at most, and one spare
            if self.runner.ensure():
                return True
            status = self.runner.status
            if status == "not_installed" and self.runner.install_prompt() is None:
                self.runner.install()   # Linux: nothing to ask; its install() sets the command as the detail
                question = None
            elif status == "not_installed":
                question, action = self.runner.install_prompt(), self.runner.install
            elif status == "no_model":
                question, action = self.runner.pull_prompt(), self.runner.pull
            else:
                question = None
            if question is None:   # Nothing to ask: the runner cannot be started from here
                if report:
                    self._emit("error", self.runner.detail)
                return False
            if not self._yes(question):
                if report:
                    self._emit("error", f"The assistant cannot answer without it: {self.runner.detail}.")
                return False
            if not action():
                if report:
                    self._emit("error", self.runner.detail)
                return False
        return False

    def _yes(self, question: str) -> bool:
        """Asks a yes-or-no question: the answer is y or n, typed or on a button."""
        reply = self._ask({"question": question, "options": ["y", "n"]})
        return str(reply.get("answer", "")).strip().lower() in ("y", "yes")

    def _run(self) -> None:
        self._busy(True)
        try:
            if not self._ready_runner(report=True):
                return
            nudges = 0
            for _ in range(MAX_CALLS):
                if windows.AGENT.stopped:
                    self._pause()
                self._set_state("thinking")
                thought: list = []   # Kept as it comes, so a reply cut off keeps its reasoning so far

                def on_reasoning(piece: str) -> None:
                    thought.append(piece)
                    self.event.emit("thought_part", piece)
                try:
                    reply = self.llm.chat(self._context(), self.tools,
                                          on_text=lambda piece: self.event.emit("part", piece),
                                          cancelled=lambda: windows.AGENT.stopped, on_reasoning=on_reasoning,
                                          max_tokens=self.settings.get("max_tokens"))
                except Cancelled:
                    if "".join(thought).strip():
                        self._emit("reasoning", "".join(thought).strip())
                    self._emit("status", "(reply cut off)")
                    self._pause()
                    continue
                except LLMError as e:
                    self._emit("error", str(e))
                    return
                if (reply.get("reasoning") or "").strip():
                    self._emit("reasoning", reply["reasoning"].strip())
                calls = reply["tool_calls"]
                message = {"role": "assistant", "content": reply["content"]}
                if calls:
                    message["tool_calls"] = [{"id": c["id"], "type": "function", "function": {
                        "name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in calls]
                self.messages.append(message)
                if reply["content"].strip():
                    self._emit("assistant", reply["content"].strip())
                if not calls:
                    said = reply["content"].lower()
                    if nudges < NUDGES and any(word in said for word in ANNOUNCES) and \
                            not any(word in said for word in OFFERS):
                        nudges += 1   # It said what it would do, and stopped: it should do it
                        self.messages.append({"role": "user", "content": NUDGE})
                        continue
                    return    # The answer: the request is done
                for k, call in enumerate(calls):
                    result = self._do(call)
                    self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": _compact(result)})
                    if "stopped" in result:
                        for later in calls[k + 1:]:   # Every call needs its answer; these were not made
                            self.messages.append({"role": "tool", "tool_call_id": later["id"],
                                                  "content": '{"skipped":"the user stopped you first"}'})
                        self._pause()
                        break
            self._emit("error", f"Stopped after {MAX_CALLS} tool calls: send a message to go on.")
        except _Ended:
            self._emit("status", "Request ended.")
        finally:
            self._busy(False)
            self._set_state("idle")

    def _do(self, call: dict) -> dict:
        name, args = call["name"], call["arguments"]
        if call.get("bad"):
            return {"error": f"the arguments of {name} were not JSON: {call['bad']}"}
        if name == "ask_user":
            return self._ask(args)
        if name not in MODEL_TOOLS:
            return {"error": f"no tool {name!r}; tools: {', '.join(MODEL_TOOLS)}, ask_user"}
        self._set_state("acting")
        shown = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())
        self._emit("tool", f"{name}({shown})")
        result = bridge.call(name, args)
        self._emit("result", _summary(name, result))
        return result

    def _ask(self, args: dict) -> dict:
        question = str(args.get("question") or "").strip() or "?"
        options = [str(o) for o in (args.get("options") or [])][:4]
        self._answer = None
        self._answered.clear()
        self._emit("ask", json.dumps({"q": question, "options": options}, ensure_ascii=False))
        self._set_state("waiting")
        self._busy(False)   # Waiting for the user is not working: Space plays the video again
        while not self._answered.wait(0.2):
            if windows.AGENT.stopped:
                break
        self._busy(True)
        if self._answer is None:
            return {"stopped": "the user stopped instead of answering"}
        return {"answer": self._answer}

    def _pause(self) -> None:
        """Stopped by the user: waits for Start (or End), then tells the model what changed."""
        self._set_state("stopped")
        self._busy(False)
        self._emit("status", "Stopped. Fix what you need, then press Start (a note typed in the box "
                             "goes with it), or End.")
        self._note, self._end = "", False
        self._go.clear()
        self._go.wait()
        changes = _payload(bridge.call("resume"))   # Lets tools run again, says what the user did
        if self._end:
            raise _Ended()
        self._busy(True)
        user_changes = changes.get("user_changes", "none") if isinstance(changes, dict) else changes
        note = self._note.strip()
        self.messages.append({"role": "user", "content": (
            "I stopped you" + (f" and say: {note}" if note else "") + ". My changes meanwhile "
            f"(history events [n, who, op, text, params]): {json.dumps(user_changes, ensure_ascii=False)[:1200]}. "
            "Take them into account and go on.")})
        self._emit("status", "Going on.")

    def _context(self) -> list:
        """The messages for the next turn: old tool results cut, oldest requests dropped."""
        messages = self.messages
        if len(messages) > MAX_MESSAGES:   # Start at a request (never between a call and its result)
            first = next((i for i in range(len(messages) - MAX_MESSAGES, len(messages))
                          if messages[i]["role"] == "user"), 0)
            messages = messages[first:]
        cut = len(messages) - KEEP_FULL
        out = []
        for i, m in enumerate(messages):
            if m["role"] == "tool" and i < cut and len(m["content"]) > OLD_RESULT_CHARS:
                m = {**m, "content": m["content"][:OLD_RESULT_CHARS] + "…"}
            out.append(m)
        return [{"role": "system", "content": SYSTEM}] + out

    # --- Small helpers -----------------------------------------------------------------------------
    def _emit(self, kind: str, text: str) -> None:
        self.transcript.append((kind, text))
        self.event.emit(kind, text)

    def _set_state(self, state: str) -> None:
        self.state = state
        self.state_changed.emit(state)

    def _busy(self, busy: bool) -> None:
        """Tells the gate whether the assistant is working (Space stops it only then)."""
        def set_busy():
            windows.AGENT.session_busy = busy
            windows.AGENT.changed.emit()
        bridge.on_gui(set_busy)


_shared: AgentSession | None = None


def shared() -> AgentSession:
    """The app's one assistant (create it on the Qt thread)."""
    global _shared
    if _shared is None:
        _shared = AgentSession()
    return _shared
