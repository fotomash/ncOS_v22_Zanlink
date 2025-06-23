<file name=agent_registry.py>
AGENT_PROFILES = {
    "Rysiek": {
        "role": "HTF Wyckoff phase specialist",
        "source": "H1/H4 Wyckoff schematic + phase confidence",
        "filters": ["AR", "ST", "Spring", "LPS", "SOS"],
        "trigger_logic": "evaluate_wyckoff_phase_context",
        "active": True
    },
    # ... other agent profiles ...
}
</file>
