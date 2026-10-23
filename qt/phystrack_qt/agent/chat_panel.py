"""
The Assistant dock, under the History: the conversation with the in-app assistant (session.py),
a box to type in, and Stop / Start. Every window has one; they all show the one conversation.

While the assistant works, a banner says that Space stops it. After Send, the keyboard goes back
to the picture, so Space does that (in the text box, Space types a space). A question from the
assistant shows its answers as buttons.
"""

import html
import json
import threading

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextBrowser,
                               QVBoxLayout, QWidget)

STATES = {"idle": "ready", "thinking": "thinking…", "acting": "working…", "waiting": "asks you",
          "stopped": "stopped"}
COLOURS = {"user": "#2b6cb0", "assistant": "#1a202c", "tool": "#6b7280", "result": "#6b7280",
           "status": "#b7791f", "error": "#c53030", "ask": "#2f855a"}


class ChatPanel(QWidget):
    _models_ready = Signal(list)

    def __init__(self, session, focus_after_send=None) -> None:
        super().__init__()
        self.session = session
        self.focus_after_send = focus_after_send   # Where the keyboard goes after Send (the picture)
        self._live = ""                            # The reply being written

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        top = QHBoxLayout()
        top.addWidget(QLabel("Model"))
        self.model_box = QComboBox()
        self.model_box.setEditable(True)
        self.model_box.addItem(session.llm.model)
        self.model_box.setToolTip(f"The local model, from the runner at {session.llm.url} "
                                  "(PHYSTRACK_LLM_URL: Ollama here; Strata or llama-server elsewhere)")
        top.addWidget(self.model_box, 1)
        self.state_label = QLabel()
        top.addWidget(self.state_label)
        layout.addLayout(top)

        self.view = QTextBrowser()
        self.view.setOpenLinks(False)
        layout.addWidget(self.view, 1)

        self.banner = QLabel("Assistant working: Space stops it")
        self.banner.setStyleSheet("QLabel { color: white; background: #3a7bd5; padding: 2px 6px; }")
        self.banner.hide()
        layout.addWidget(self.banner)

        self.options = QHBoxLayout()   # A question's answers, as buttons
        layout.addLayout(self.options)

        bottom = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Ask the assistant (e.g. do what I did in w1 on trials 2 to 4)")
        self.input.returnPressed.connect(self._send)
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self._send)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_or_start)
        self.end_btn = QPushButton("End")
        self.end_btn.setToolTip("Drop the stopped request")
        self.end_btn.clicked.connect(session.end)
        for w in (self.input, self.send_btn, self.stop_btn, self.end_btn):
            bottom.addWidget(w, 1 if w is self.input else 0)
        layout.addLayout(bottom)

        session.event.connect(self._on_event)
        session.state_changed.connect(self._on_state)
        self._models_ready.connect(self._fill_models)
        self._render()
        self._on_state(session.state)
        threading.Thread(target=self._load_models, daemon=True).start()

    # --- Actions -----------------------------------------------------------------------------------
    def _send(self) -> None:
        text = self.input.text()
        self.input.clear()
        model = self.model_box.currentText().strip()
        if model and self.session.state == "idle":
            self.session.llm.model = model
        self.session.send(text)
        if self.focus_after_send is not None and self.session.state != "waiting":
            self.focus_after_send.setFocus()   # So Space stops the assistant, as the banner says

    def _stop_or_start(self) -> None:
        if self.session.state == "stopped":
            self._send()
        else:
            self.session.stop()

    def _answer(self, text: str) -> None:
        self.session.send(text)

    # --- From the session --------------------------------------------------------------------------
    def _on_event(self, kind: str, text: str) -> None:
        if kind == "part":
            self._live += text
        else:
            if kind == "assistant" or kind in ("tool", "error", "status"):
                self._live = ""
            if kind == "ask":
                self._show_options(json.loads(text).get("options", []))
            elif kind in ("user", "status", "error"):
                self._show_options([])
        self._render()

    def _on_state(self, state: str) -> None:
        self.state_label.setText(STATES.get(state, state))
        working = state in ("thinking", "acting")
        self.banner.setVisible(working)
        self.stop_btn.setText("Start" if state == "stopped" else "Stop")
        self.stop_btn.setEnabled(state != "idle")
        self.end_btn.setVisible(state == "stopped")
        self.send_btn.setEnabled(state in ("idle", "waiting", "stopped"))
        self.model_box.setEnabled(state == "idle")
        if state == "idle":
            self._live = ""
            self._show_options([])
            self._render()

    def _show_options(self, options: list) -> None:
        while self.options.count():
            item = self.options.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        for option in options:
            button = QPushButton(option)
            button.clicked.connect(lambda _=False, o=option: self._answer(o))
            self.options.addWidget(button)

    def _render(self) -> None:
        rows = []
        for kind, text in self.session.transcript:
            colour = COLOURS.get(kind, "#1a202c")
            if kind == "ask":
                text = json.loads(text).get("q", "")
            body = html.escape(text).replace("\n", "<br>")
            prefix = {"user": "<b>You:</b> ", "assistant": "", "tool": "▸ ", "result": "&nbsp;&nbsp;",
                      "ask": "<b>?</b> ", "status": "", "error": "<b>!</b> "}.get(kind, "")
            size = " font-size: small;" if kind in ("tool", "result") else ""
            rows.append(f'<p style="margin: 2px 0; color: {colour};{size}">{prefix}{body}</p>')
        if self._live:
            rows.append(f'<p style="margin: 2px 0; color: {COLOURS["assistant"]};">'
                        f'{html.escape(self._live)}▍</p>')
        self.view.setHtml("".join(rows))
        bar = self.view.verticalScrollBar()
        bar.setValue(bar.maximum())

    # --- Models ------------------------------------------------------------------------------------
    def _load_models(self) -> None:
        try:
            self._models_ready.emit(self.session.llm.models())
        except Exception:   # The runner is not up yet: it starts with the first request
            pass

    def _fill_models(self, models: list) -> None:
        current = self.model_box.currentText()
        self.model_box.blockSignals(True)
        self.model_box.clear()
        self.model_box.addItems(models if current in models else [current] + models)
        self.model_box.setCurrentText(current)
        self.model_box.blockSignals(False)
