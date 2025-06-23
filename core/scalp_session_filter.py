"""
scalp_session_filter.py

Session-based trading filter for ZANALYTICS scalper module.
Prevents entries during known low-liquidity times (e.g., rollover, post-settlement)
and activates only during predefined high-probability sessions (e.g., London, NY).

Returns a structured dict:
{
    "session_ok": True/False,
    "session_name": "London" or None,
    "utc_time": "13:45",
    "reason": "Inside London window (07:00–10:30)"
}
"""

from datetime import datetime, time

def is_within_session_time(current_utc: datetime, config: dict) -> dict:
    """
    Check if the current UTC time is within allowed trading sessions and outside of blocked windows.
    Returns a dict with session_ok flag and reason.
    """

    current_time = current_utc.time()
    session_ok = False
    active_session = None
    reason = ""

    # Parse allowed sessions
    for session in config.get("session_blocks", []):
        start = datetime.strptime(session["start"], "%H:%M").time()
        end = datetime.strptime(session["end"], "%H:%M").time()
        if start <= current_time <= end:
            session_ok = True
            active_session = session["name"]
            reason = f"Inside {active_session} window ({session['start']}–{session['end']})"
            break

    # Check blocked windows override
    for blocked in config.get("blocked_windows", []):
        block_start = datetime.strptime(blocked["start"], "%H:%M").time()
        block_end = datetime.strptime(blocked["end"], "%H:%M").time()
        if block_start <= current_time <= block_end:
            session_ok = False
            reason = f"Blocked window: {blocked['start']}–{blocked['end']}"
            break

    return {
        "session_ok": session_ok,
        "session_name": active_session if session_ok else None,
        "utc_time": current_time.strftime("%H:%M"),
        "reason": reason or "Outside allowed session windows"
    }