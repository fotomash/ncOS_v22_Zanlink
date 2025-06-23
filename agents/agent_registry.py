# agent_registry.py

AGENT_PROFILES = {
    "Bożenka": {
        "role": "signal validator",
        "source": "M1/M5 Wyckoff CHoCH/Spring/BOS",
        "filters": ["spread", "tick_ret", "volume"],
        "trigger_logic": "evaluate_microstructure_phase_trigger",
        "active": True
    },
    "Stefania": {
        "role": "reputation auditor / trust scoring",
        "source": "past entries + score feedback",
        "filters": ["phase", "trust_score", "journal"],
        "trigger_logic": "score_contextual_trust",
        "active": True  # can be enabled by config
    },
    "Lusia": {
        "role": "semantic confluence engine",
        "source": "indicators + DSS + divergence + config tags",
        "filters": ["DSS slope", "EMA polarity", "BB compression"],
        "trigger_logic": "evaluate_indicator_alignment",
        "active": True
    },
    "Zdzisiek": {
        "role": "compliance/risk monitor",
        "source": "spread + volatility + tick clusters",
        "filters": ["ATR", "spread", "max risk"],
        "trigger_logic": "evaluate_risk_profile",
        "active": True
    },
    "Rysiek": {
        "role": "HTF phase tracker",
        "source": "H1/H4 Wyckoff schematic + phase confidence",
        "filters": ["AR", "ST", "Spring", "LPS", "SOS"],
        "trigger_logic": "evaluate_wyckoff_phase_context",
        "active": True
    }
}


def list_active_agents():
    return [name for name, profile in AGENT_PROFILES.items() if profile.get("active")]


def get_agent_profile(name):
    return AGENT_PROFILES.get(name, None)
