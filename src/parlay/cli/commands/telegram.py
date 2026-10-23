"""``parlay telegram``: interactive bot plus lineup-window alerts."""

from __future__ import annotations

import argparse
import os

from parlay.cli._common import add_database


def register(commands: argparse._SubParsersAction) -> None:
    bot = commands.add_parser("telegram", help="run the Telegram bot (commands + watcher alerts)")
    add_database(bot)
    bot.add_argument(
        "--setup", action="store_true", help="detect your chat id (send /start to the bot first)"
    )
    bot.add_argument("--no-watch", action="store_true", help="do not run the lineup-window watcher")
    bot.add_argument("--bankroll", type=float, default=1000.0)
    bot.set_defaults(handler=run_telegram)


def run_telegram(args: argparse.Namespace) -> int:
    from parlay.config import env_file_path, set_env_value
    from parlay.telegram_bot import HELP, ParlayBot, TelegramAPI, find_chat_ids, run_bot

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit(f"Set TELEGRAM_BOT_TOKEN (from @BotFather), e.g. in {env_file_path()}")
    api = TelegramAPI(token)
    me = api.get_me()
    username = me.get("username", "your bot")

    if args.setup:
        chats = find_chat_ids(api)
        if not chats:
            print(f"No messages yet. Open Telegram, send /start to @{username}, then run this again.")
            return 1
        chat_id, name = chats[-1]
        path = set_env_value("TELEGRAM_CHAT_ID", chat_id)
        api.send_message(chat_id, f"✅ Connected to parlay. Hi {name}!\n\n{HELP}")
        print(f"Saved TELEGRAM_CHAT_ID={chat_id} ({name}) to {path}; a welcome message was sent.")
        if len(chats) > 1:
            print(
                "Other chats that messaged the bot (not authorized): " + ", ".join(c for c, _ in chats[:-1])
            )
        return 0

    allowed = {c.strip() for c in os.environ.get("TELEGRAM_CHAT_ID", "").split(",") if c.strip()}
    if not allowed:
        raise SystemExit("TELEGRAM_CHAT_ID is not set; run `parlay telegram --setup` first")
    bot = ParlayBot(
        api, allowed, database=args.database, books=os.environ.get("PARLAY_BOOKS"), bankroll=args.bankroll
    )
    watcher = None
    if not args.no_watch and os.environ.get("THE_ODDS_API_KEY"):
        from parlay.betting.value import ValueConfig
        from parlay.betting.watch import WatchConfig, Watcher
        from parlay.data.database import ResearchDatabase
        from parlay.notify import TelegramNotifier

        books = os.environ.get("PARLAY_BOOKS")
        config = WatchConfig(
            bankroll=args.bankroll,
            value=ValueConfig(bookmakers=tuple(b.strip() for b in books.split(",")) if books else None),
        )
        notifier = TelegramNotifier(token, sorted(allowed)[0])
        watcher = Watcher(
            ResearchDatabase(args.database), os.environ["THE_ODDS_API_KEY"], config, notifier=notifier
        )
    print(
        f"@{username} is running for chat(s) {', '.join(sorted(allowed))}; "
        f"watcher: {'on' if watcher else 'off'}."
    )
    for chat in allowed:
        api.send_message(chat, f"🟢 parlay bot online · watcher {'on' if watcher else 'off'} · /help")
    try:
        run_bot(bot, watcher=watcher)
    except KeyboardInterrupt:
        print("Stopped.")
    return 0
