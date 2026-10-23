"""
The in-app assistant (agent/session.py, agent/chat_panel.py), with a scripted model instead of a
real one: a request runs tools and ends with an answer; a question waits for the user's answer;
Space stops it while it thinks, and Start tells the model what the user changed meanwhile.

    ~/.venvs/phystrackx-qt/bin/python tests/test_assistant.py      (PHYSTRACK_QT_SHOW=1 to watch)
"""

import json
import time

import harness as h  # noqa: F401  (sets up Qt before PySide6 is imported)

from phystrack_qt import windows  # noqa: E402
from phystrack_qt.agent.chat_panel import ChatPanel  # noqa: E402
from phystrack_qt.agent.llm import Cancelled  # noqa: E402
from phystrack_qt.agent.session import AgentSession  # noqa: E402
from test_tracking import make_clip  # noqa: E402


class ScriptedModel:
    """Plays the model: each turn is a dict {"say": text, "calls": [(tool, args)]}, or a function
    (messages, on_text, cancelled) -> reply (to look at what it was sent, or to be slow)."""

    def __init__(self, turns: list) -> None:
        self.turns, self.sent = list(turns), []
        self.url, self.model, self.is_ollama = "scripted", "scripted", False

    def models(self) -> list:
        return ["scripted"]

    def chat(self, messages, tools, on_text=None, cancelled=None, max_tokens=400) -> dict:
        self.sent.append(messages)
        turn = self.turns.pop(0)
        if callable(turn):
            return turn(messages, on_text, cancelled)
        if turn.get("say") and on_text:
            on_text(turn["say"])
        return {"content": turn.get("say", ""), "tool_calls": [
            {"id": f"c{len(self.sent)}{i}", "name": name, "arguments": args}
            for i, (name, args) in enumerate(turn.get("calls", []))]}


def run_until(session, states, timeout=30.0) -> bool:
    return h.pump(lambda: session.state in states, timeout)


def test_assistant() -> None:
    w = h.window()
    w.load_video(make_clip())
    h.pump(lambda: w.source.has_proxy, 60)

    # A request: a tool call, then the answer
    model = ScriptedModel([{"calls": [("windows", {})]}, {"say": f"One window is open: {w.wid}."}])
    session = AgentSession(llm=model)
    panel = ChatPanel(session)
    session.send("Which windows are open?")
    h.check("a request ends", h.pump(lambda: session.state == "idle" and len(model.sent) == 2, 30))
    tool_result = model.sent[1][-1]
    h.check("the model gets the tool's result", tool_result["role"] == "tool" and w.wid in tool_result["content"],
            str(tool_result)[:200])
    h.check("the system prompt and the tool list are sent each turn",
            model.sent[0][0]["role"] == "system" and model.sent[1][0] == model.sent[0][0])
    text = panel.view.toPlainText()
    h.check("the dock shows the request, the tool call and the answer",
            "Which windows" in text and "windows()" in text and f"One window is open: {w.wid}" in text, text[-300:])

    # A question: the session waits for the answer, which goes back to the model
    def check_answer(messages, on_text, cancelled):
        answer = json.loads(messages[-1]["content"])
        return {"content": f"You said {answer.get('answer')}.", "tool_calls": []}
    model.turns = [{"calls": [("ask_user", {"question": "Which object?", "options": ["1", "2"]})]}, check_answer]
    session.send("Track the ball")
    h.check("a question waits for the user", run_until(session, ("waiting",)))
    h.check("the question's answers are buttons", panel.options.count() == 2)
    panel.options.itemAt(1).widget().click()
    h.check("the answer goes back to the model", h.pump(lambda: session.state == "idle", 30) and
            session.transcript[-1] == ("assistant", "You said 2."), str(session.transcript[-2:]))

    # Space while it thinks: the reply is cut off; the user changes the trim; Start goes on,
    # and the model is told about the change and the note
    def slow(messages, on_text, cancelled):
        while not cancelled():
            time.sleep(0.02)
        raise Cancelled()

    def told(messages, on_text, cancelled):
        told.message = messages[-1]["content"]
        return {"content": "Going on with your trim.", "tool_calls": []}
    model.turns = [slow, told]
    session.send("Do something slow")
    h.check("it is thinking", run_until(session, ("thinking",)))
    h.pump(lambda: windows.AGENT.session_busy, 5)
    w.activateWindow()
    h.pump(timeout=0.2)
    w._space()
    h.check("Space stops it while it thinks", run_until(session, ("stopped",)))
    h.check("while stopped, Start is offered", panel.stop_btn.text() == "Start")
    w._trim_marks = [3, 60]
    w._apply_trim()
    h.pump(timeout=0.5)
    panel.input.setText("I fixed the trim")
    panel.stop_btn.click()   # Start, with the note
    h.check("Start goes on to the end", h.pump(lambda: session.state == "idle", 30))
    message = getattr(told, "message", "")
    h.check("the model is told what the user changed, and their note",
            "trim" in message and "I fixed the trim" in message, message[:300])
    h.check("the gate is free again", not windows.AGENT.stopped and not windows.AGENT.session_busy)

    # A tool error goes back to the model as an error, and an unknown tool is refused
    model.turns = [{"calls": [("trim", {"start": 1})]}, {"calls": [("plots", {"kinds": ["nope"]})]},
                   {"say": "Done."}]
    session.send("Break things")
    h.pump(lambda: session.state == "idle" and not model.turns, 30)
    results = [m["content"] for m in session.messages if m["role"] == "tool"][-2:]
    h.check("errors and unknown tools are reported to the model, not raised",
            "no tool" in results[0] and "unknown kinds" in results[1], str(results))

    # A click anywhere else takes the text box's selection away (not only a click on the video)
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication
    panel.input.setText("a draft")
    panel.input.selectAll()
    h.check("the text box has a selection to clear", panel.input.hasSelectedText())
    press = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(2, 2), QPointF(2, 2), Qt.MouseButton.LeftButton,
                        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(panel.send_btn, press)
    h.check("a click on a button elsewhere clears the selection", not panel.input.hasSelectedText())
    panel.input.selectAll()
    QApplication.sendEvent(panel.view.viewport(), press)
    h.check("a click on the history clears the selection too", not panel.input.hasSelectedText())
    w._dirty = False
    w.close()


if __name__ == "__main__":
    h.app()
    test_assistant()
    h.finish()
