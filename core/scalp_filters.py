from microstructure_filter import evaluate_microstructure
import pandas as pd
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


def check_session_filter(config: dict) -> dict:
    """
    Wrapper that checks session time validity using current UTC time.
    Can be extended with logging or override modes later.
    """
    current_utc = datetime.utcnow()
    session_info = is_within_session_time(current_utc, config)

    if not session_info["session_ok"]:
        print(f"[SCALP SESSION FILTER] ❌ Entry blocked at {session_info['utc_time']} — {session_info['reason']}")
    else:
        print(f"[SCALP SESSION FILTER] ✅ Session OK — {session_info['reason']}")

    return session_info


# --- Microstructure scalp validation ---
def validate_scalp_signal(signal, ticks_df, config):
    """
    Validate a ScalpSignal using microstructure logic and optional confluence checks.
    Returns confirmation status and metadata.
    """
    rr = signal.get("rr", None)
    micro = evaluate_microstructure(
        ticks_df,
        window=config.get("tick_window", 5),
        min_drift_pips=config.get("min_drift_pips", 1.0),
        max_spread=config.get("max_spread", 1.8),
        min_rr=config.get("min_rr_threshold", 1.8),
        rr=rr
    )

    if not micro.get("micro_confirmed", False):
        return {"confirmed": False, "reason": micro.get("reason", "Microstructure failed")}

    return {"confirmed": True, "metadata": micro}