"""Minimal Telegram Bot API client: long polling, messages with inline buttons."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

MAX_MESSAGE = 3800
Buttons = list[list[tuple[str, str]]]  # rows of (label, callback data)


def keyboard(buttons: Buttons | None) -> str | None:
    if not buttons:
        return None
    rows = [[{"text": label, "callback_data": data[:64]} for label, data in row] for row in buttons]
    return json.dumps({"inline_keyboard": rows})


class TelegramAPI:
    def __init__(self, token: str, *, timeout: int = 40):
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")
        self.base = f"https://api.telegram.org/bot{token}"
        self.timeout = timeout

    def _call(self, method: str, params: dict[str, Any], *, timeout: int | None = None) -> Any:
        data = urlencode({k: v for k, v in params.items() if v is not None}).encode()
        with urlopen(
            Request(f"{self.base}/{method}", data=data), timeout=timeout or self.timeout
        ) as response:
            payload = json.load(response)
        if not payload.get("ok"):
            raise OSError(f"Telegram {method} failed: {payload.get('description')}")
        return payload["result"]

    def get_me(self) -> dict[str, Any]:
        return self._call("getMe", {})

    def get_updates(self, offset: int | None = None, timeout: int = 25) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "timeout": timeout,
            "allowed_updates": json.dumps(["message", "callback_query"]),
        }
        if offset is not None:
            params["offset"] = offset
        return self._call("getUpdates", params, timeout=timeout + 15)

    def send_message(self, chat_id: int | str, text: str, *, buttons: Buttons | None = None) -> None:
        chunks = split_message(text)
        for i, chunk in enumerate(chunks):
            self._call(
                "sendMessage",
                {
                    "chat_id": chat_id,
                    "text": chunk,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": "true",
                    "reply_markup": keyboard(buttons) if i == len(chunks) - 1 else None,
                },
            )

    def answer_callback(self, callback_id: str, text: str | None = None) -> None:
        self._call("answerCallbackQuery", {"callback_query_id": callback_id, "text": text})

    def set_commands(self, commands: list[tuple[str, str]]) -> None:
        self._call(
            "setMyCommands", {"commands": json.dumps([{"command": c, "description": d} for c, d in commands])}
        )


def split_message(text: str, limit: int = MAX_MESSAGE) -> list[str]:
    chunks, current = [], ""
    for line in text.splitlines() or [""]:
        if current and len(current) + len(line) + 1 > limit:
            chunks.append(current)
            current = ""
        current = f"{current}\n{line}" if current else line
    chunks.append(current)
    return [c for c in chunks if c.strip()] or [text]
