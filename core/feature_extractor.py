def extract_predictive_features(config, tf_data, wyckoff_result) -> dict:
    """
    Extracts and normalizes predictive features from the input data sources.
    Expected output range: [0.0, 1.0] for each feature.
    """
    return {
        "htf_bias": 1.0 if config.get("context_bias", "bullish") == "bullish" else -1.0,
        "idm_detected": 1.0 if wyckoff_result.get("inducement", False) else 0.0,
        "sweep_validated": wyckoff_result.get("sweep_confidence", 0.5),
        "choch_confirmed": wyckoff_result.get("choch_strength", 0.6),
        "poi_validated": wyckoff_result.get("poi_score", 0.7),
        "tick_density": tf_data.get("tick_density", 0.5),
        "spread_status": 1.0 - tf_data.get("spread_instability", 0.5)
    }
