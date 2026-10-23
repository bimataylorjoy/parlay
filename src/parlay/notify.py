"""Optional notifications (Telegram bot) for value bets found by `parlay watch`."""

from __future__ import annotations

import json
import os
from typing import Protocol
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class Notifier(Protocol):
    def send(self, text: str) -> None: ...


class TelegramNotifier:
    """Send messages through a Telegram bot (create one with @BotFather; free)."""

    def __init__(self, token: str, chat_id: str, *, timeout: int = 15):
        if not token or not chat_id:
            raise ValueError("Telegram token and chat id are required")
        self.token = token
        self.chat_id = chat_id
        self.timeout = timeout

    def send(self, text: str) -> None:
        data = urlencode(
            {"chat_id": self.chat_id, "text": text[:4000], "disable_web_page_preview": "true"}
        ).encode()
        request = Request(f"https://api.telegram.org/bot{self.token}/sendMessage", data=data)
        with urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        if not payload.get("ok"):
            raise OSError(f"Telegram rejected the message: {payload}")


class PrintNotifier:
    def send(self, text: str) -> None:
        print(text)


def notifier_from_env() -> Notifier | None:
    """A Telegram notifier when ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID`` are set."""
    token, chat_id = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    return TelegramNotifier(token, chat_id) if token and chat_id else None
