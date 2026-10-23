"""Beginner-friendly Indonesian messages for the Telegram bot."""

from __future__ import annotations

import html
import math
import os
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from parlay.betting.tickets import Ticket
from parlay.betting.value import ValueBet
from parlay.data.odds_columns import parse_market

DAYS = ("Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu")
MONTHS = ("Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des")
COMPETITIONS = {"EPL": "Liga Inggris", "Championship": "Championship Inggris"}


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def min_stake() -> float:
    return float(os.environ.get("PARLAY_MIN_STAKE", 5000))


def edge_label(ev: float) -> str:
    if ev >= 0.06:
        return "🟢 Edge besar (cek dulu odds di bookie masih sama)"
    if ev >= 0.03:
        return "🟡 Edge sedang"
    return "⚪ Edge tipis"


def stake_line(stake: float, fraction: float) -> str:
    share = pct_small(fraction)
    if stake < min_stake():
        return (
            f"💵 Saran taruhan: <b>{money(stake)}</b> ({share} modal) — di bawah minimal taruhan "
            f"{money(min_stake())}. Edge-nya tipis: pakai minimal saja atau cukup 🧪 Simulasi."
        )
    return f"💵 Saran taruhan: <b>{money(stake)}</b> ({share} modal)"


def pct_small(value: float) -> str:
    return (f"{value:.2%}" if value < 0.01 else f"{value:.1%}").replace(".", ",")


def local_zone() -> Any:
    try:
        return ZoneInfo(os.environ.get("PARLAY_TIMEZONE", "Asia/Jakarta"))
    except ZoneInfoNotFoundError:  # pragma: no cover - systems without tz data
        return timezone.utc


def zone_label(moment: datetime) -> str:
    name = moment.tzname() or "UTC"
    return {"+07": "WIB", "+08": "WITA", "+09": "WIT"}.get(name, name)


def when(moment: datetime | None) -> str:
    if moment is None:
        return "jam belum diketahui"
    local = moment.astimezone(local_zone())
    return f"{DAYS[local.weekday()]} {local.day} {MONTHS[local.month - 1]}, {local:%H:%M} {zone_label(local)}"


def money(amount: float, currency: str | None = None) -> str:
    currency = currency if currency is not None else os.environ.get("PARLAY_CURRENCY", "Rp")
    sign = "-" if amount < 0 else ""
    rounded = round(abs(amount))
    text = f"{rounded:,}".replace(",", ".") if rounded >= 1 else f"{abs(amount):.2f}".replace(".", ",")
    return f"{sign}{currency}{text}"


def pct(value: float, signed: bool = False) -> str:
    return f"{value:+.1%}".replace(".", ",") if signed else f"{value:.1%}".replace(".", ",")


def _line(value: float) -> str:
    return f"{value + 0.0:+g}".replace(".", ",")


def _goals_rule(selection: str, line: float) -> str:
    over = selection == "over"
    if (line * 2) % 1 == 0 and line % 1:  # x.5
        target = math.ceil(line) if over else math.floor(line)
        return f"menang jika total gol {'≥' if over else '≤'} {target}"
    if line % 1 == 0:
        target = int(line) + 1 if over else int(line) - 1
        return f"menang jika total gol {'≥' if over else '≤'} {target}, uang kembali jika tepat {int(line)}"
    return "garis seperempat: bisa menang/kalah setengah taruhan"


def _handicap_rule(line: float) -> str:
    if line == -0.5:
        return "menang jika tim ini menang"
    if line == 0.5:
        return "menang jika tim ini tidak kalah (menang atau seri)"
    if (line * 2) % 1 == 0 and line % 1:
        if line < 0:
            return f"menang jika tim ini menang dengan selisih ≥ {math.ceil(-line)} gol"
        return f"menang jika tim ini kalah maksimal {math.floor(line)} gol (atau tidak kalah)"
    if line == 0:
        return "menang jika tim ini menang, uang kembali jika seri"
    return "handicap seperempat/bulat: bisa uang kembali atau menang/kalah setengah"


def describe_bet(home: str, away: str, market: str, selection: str) -> tuple[str, str]:
    """``(pilihan, penjelasan pasar)`` in plain Indonesian."""
    kind, line = parse_market(market)
    if kind == "1x2":
        pick = {"home": f"Menang {home}", "away": f"Menang {away}", "draw": "Seri"}[selection]
        return pick, "1X2 · hasil akhir 90 menit"
    assert line is not None
    if kind == "totals":
        number = f"{line:g}".replace(".", ",")
        pick = f"{'Over' if selection == 'over' else 'Under'} {number} gol"
        return pick, f"Total gol · {_goals_rule(selection, line)}"
    team, team_line = (home, line) if selection == "home" else (away, -line)
    return f"{team} {_line(team_line)}", f"Handicap Asia · {_handicap_rule(team_line)}"


def value_card(bet: ValueBet, bankroll: float) -> str:
    pick, market = describe_bet(bet.home_team, bet.away_team, bet.market, bet.selection)
    stake = bet.stake_fraction * bankroll
    per_thousand = bet.expected_value * 1000
    competition = COMPETITIONS.get(bet.competition, bet.competition)
    return (
        f"⚽ <b>{esc(bet.home_team)} vs {esc(bet.away_team)}</b>"
        f" · {esc(competition)}\n"
        f"🕒 {when(bet.kickoff_at)}\n\n"
        f"👉 <b>Pasang: {esc(pick)}</b>\n"
        f"     {esc(market)}\n"
        f"💰 Odds {esc(bet.bookmaker)}: <b>{bet.odds:.2f}</b>  (wajar {bet.fair_odds:.2f})\n"
        f"🎲 Peluang menang: <b>{pct(bet.fair_probability)}</b>\n"
        f"📈 EV: <b>{pct(bet.expected_value, signed=True)}</b> → rata-rata untung {money(per_thousand)} "
        f"per {money(1000)} dipasang\n"
        f"{stake_line(stake, bet.stake_fraction)}\n"
        f"{edge_label(bet.expected_value)}"
    )


def ticket_card(ticket: Ticket, bankroll: float) -> str:
    lines = [f"🎟 <b>Mix parlay {len(ticket.legs)} tim</b> · {esc(ticket.bookmaker)}"]
    for number, leg in enumerate(ticket.legs, start=1):
        pick, _ = describe_bet(leg.home_team, leg.away_team, leg.market, leg.selection)
        leg_ev = f" · EV {pct(leg.expected_value, signed=True)}" if leg.expected_value is not None else ""
        warn = " ⚠️" if leg.expected_value is not None and leg.expected_value < 0 else ""
        lines.append(
            f"{number}. {esc(leg.home_team)} vs {esc(leg.away_team)}: "
            f"<b>{esc(pick)}</b> @ {leg.odds:.2f}{leg_ev}{warn}"
        )
    boost = f" (termasuk boost {pct(ticket.boost)})" if ticket.boost else ""
    lines.append(f"\n💰 Total odds: <b>{ticket.combined_odds:.2f}</b>{boost}")
    p = ticket.win_probability
    if p is None:
        lines.append("❔ Ada tim tanpa harga Pinnacle, EV tidak bisa dihitung.")
        return "\n".join(lines)
    ev = ticket.expected_value or 0.0
    stake = ticket.stake_fraction() * bankroll
    lines += [
        f"🎲 Peluang tiket menang: <b>{pct(p)}</b> (sekitar 1 dari {max(1, round(1 / p))} tiket)",
        f"📈 EV tiket: <b>{pct(ev, signed=True)}</b>",
    ]
    if ev <= 0:
        bad = sum(1 for leg in ticket.legs if (leg.expected_value or 0) < 0)
        lines.append(
            f"\n⛔ <b>Jangan dipasang.</b> {bad} tim bertanda ⚠️ punya odds di bawah harga wajar; "
            "di parlay kerugian kecil itu berlipat."
        )
    else:
        lines.append(
            f"💵 Saran taruhan: <b>{money(stake)}</b> — kecil saja, tiket parlay lebih sering kalah."
        )
    return "\n".join(lines)


def report_text(report: dict[str, Any], paper: dict[str, Any]) -> str:
    def block(title: str, r: dict[str, Any]) -> list[str]:
        if not r["bets"]:
            return [f"<b>{title}</b>: belum ada taruhan."]
        settled = r["settled"]
        lines = [
            f"<b>{title}</b>: {r['bets']} taruhan · {settled} selesai · {r['open']} masih berjalan",
        ]
        if settled:
            lines.append(
                f"   Untung/rugi: <b>{money(r['profit'])}</b> dari {money(r['staked'])} dipasang "
                f"({pct(r['yield'], signed=True)})"
            )
        if r["clv_n"]:
            mood = "👍 lebih bagus" if r["mean_clv"] > 0 else "👎 lebih buruk"
            lines.append(
                f"   Kualitas harga (CLV): <b>{pct(r['mean_clv'], signed=True)}</b> → odds yang diambil "
                f"{mood} dari harga penutupan"
            )
        return lines

    lines = ["📊 <b>Laporan taruhan</b>", ""]
    lines += block("Taruhan asli", report)
    lines += ["", *block("Simulasi (paper)", paper)]
    parlays = report.get("parlays") or {}
    if parlays.get("tickets"):
        lines += [
            "",
            f"<b>Parlay</b>: {parlays['tickets']} tiket · {parlays['won']} menang · "
            f"{parlays['open']} berjalan · untung/rugi {money(parlays['profit'])}",
        ]
    lines += ["", f"🧭 {verdict_text(report if report['clv_n'] else paper)}"]
    lines.append("\n<i>CLV positif secara konsisten adalah tanda paling awal bahwa strategi punya edge.</i>")
    return "\n".join(lines)


def verdict_text(report: dict[str, Any]) -> str:
    n = report["clv_n"]
    if n < 50:
        return f"Data belum cukup ({n}/50 taruhan dengan harga penutupan). Lanjutkan simulasi dulu."
    if report["mean_clv"] + 2 * report["clv_se"] < 0:
        return "STOP: harga yang diambil konsisten lebih buruk dari harga wajar. Jangan pakai uang asli."
    if n >= 200 and report["clv_t"] > 2:
        return "Ada bukti edge: harga yang diambil konsisten lebih bagus dari harga penutupan."
    return "Belum meyakinkan: CLV belum berbeda jelas dari nol. Lanjutkan dengan taruhan kecil/simulasi."


WELCOME = (
    "👋 <b>Halo! Saya pencari value bet.</b>\n\n"
    "Saya membandingkan odds bookie kamu dengan <b>harga wajar Pinnacle</b> (pasar paling tajam). "
    "Kalau odds bookie lebih tinggi dari harga wajar, taruhan itu punya <b>EV positif</b>: "
    "dalam jangka panjang rata-rata untung.\n\n"
    "🎲 <b>Peluang menang</b> = seberapa sering taruhan ini diperkirakan menang.\n"
    "📈 <b>EV</b> = untung rata-rata per taruhan dalam jangka panjang (bukan jaminan per taruhan).\n"
    "💵 <b>Saran taruhan</b> dihitung dari modalmu, sengaja kecil supaya aman.\n\n"
    "Pilih menu di bawah 👇"
)

HELP = (
    "ℹ️ <b>Cara pakai</b>\n"
    "• <b>🎯 Value bet</b>: daftar taruhan dengan EV positif. Tekan <b>✅ Saya pasang</b> setelah kamu "
    "memasangnya di bookie, supaya saya bisa mencatat hasilnya.\n"
    "• <b>🧪 Simulasi</b>: catat tanpa uang asli, untuk latihan.\n"
    "• <b>➕ Parlay</b>: kumpulkan beberapa value bet jadi mix parlay, lalu cek EV-nya di "
    "<b>🎟 Slip parlay</b>.\n"
    "• <b>📊 Laporan</b>: untung/rugi dan kualitas harga.\n"
    "• <b>⚙️ Modal</b>: atur modalmu supaya saran taruhan dalam rupiah tepat.\n\n"
    "Saya juga otomatis mengecek ulang harga ±50 menit sebelum kickoff (setelah susunan pemain keluar) "
    "dan mengirim pesan kalau ada value bet baru."
)
