"""``parlay telegram``: the button-driven bot plus lineup-window alerts."""

from __future__ import annotations

import argparse
import os

from parlay.cli._common import add_database


def register(commands: argparse._SubParsersAction) -> None:
    bot = commands.add_parser("telegram", help="run the Telegram bot (buttons, alerts, daily maintenance)")
    add_database(bot)
    bot.add_argument(
        "--setup", action="store_true", help="detect your chat id (send /start to the bot first)"
    )
    bot.add_argument("--no-watch", action="store_true", help="do not run the lineup-window watcher")
    bot.set_defaults(handler=run_telegram)


def find_chat_ids(api) -> list[tuple[str, str]]:
    seen: dict[str, str] = {}
    for update in api.get_updates(timeout=0):
        chat = (update.get("message") or {}).get("chat") or {}
        if chat.get("type") == "private":
            seen[str(chat["id"])] = chat.get("username") or chat.get("first_name") or ""
    return list(seen.items())


def run_telegram(args: argparse.Namespace) -> int:
    from parlay.config import env_file_path, set_env_value
    from parlay.telegram import api as telegram_api
    from parlay.telegram.bot import COMMANDS, MENU, BotNotifier, ParlayBot, run_bot
    from parlay.telegram.texts import WELCOME

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit(f"Set TELEGRAM_BOT_TOKEN (from @BotFather), e.g. in {env_file_path()}")
    api = telegram_api.TelegramAPI(token)
    username = api.get_me().get("username", "your bot")

    if args.setup:
        chats = find_chat_ids(api)
        if not chats:
            print(f"No messages yet. Open Telegram, send /start to @{username}, then run this again.")
            return 1
        chat_id, name = chats[-1]
        path = set_env_value("TELEGRAM_CHAT_ID", chat_id)
        api.send_message(chat_id, WELCOME, buttons=MENU)
        print(f"Saved TELEGRAM_CHAT_ID={chat_id} ({name}) to {path}; a welcome message was sent.")
        return 0

    allowed = {c.strip() for c in os.environ.get("TELEGRAM_CHAT_ID", "").split(",") if c.strip()}
    if not allowed:
        raise SystemExit("TELEGRAM_CHAT_ID is not set; run `parlay telegram --setup` first")
    books = os.environ.get("PARLAY_BOOKS")
    bot = ParlayBot(api, allowed, database=args.database, books=books)
    api.set_commands(COMMANDS)
    watcher = None
    if not args.no_watch and os.environ.get("THE_ODDS_API_KEY"):
        from parlay.betting.value import ValueConfig
        from parlay.betting.watch import WatchConfig, Watcher

        config = WatchConfig(value=ValueConfig(bookmakers=bot.books))
        watcher = Watcher(bot.db, os.environ["THE_ODDS_API_KEY"], config, notifier=BotNotifier(bot))
    print(
        f"@{username} running for chat(s) {', '.join(sorted(allowed))}; "
        f"watcher: {'on' if watcher else 'off'}."
    )
    for chat in allowed:
        bot.menu(chat, "🟢 Bot aktif. Pilih menu 👇")
    try:
        run_bot(bot, watcher=watcher)
    except KeyboardInterrupt:
        print("Stopped.")
    return 0
