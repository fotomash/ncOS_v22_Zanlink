import yaml
from pathlib import Path
import argparse

REQUIRED_KEYS = {
    "predictive_scoring": {
        "enabled": bool,
        "min_score_to_emit": float,
        "grade_thresholds": {
            "A": float,
            "B": float,
            "C": float
        },
        "factor_weights": dict
    },
    "spread_settings": {
        "enabled": bool,
        "window_size": int,
        "high_vol_baseline": float
    }
}

def validate_section(section_data, expected_keys, parent_path="") -> list:
    errors = []
    for key, expected_type in expected_keys.items():
        full_path = f"{parent_path}.{key}" if parent_path else key
        if key not in section_data:
            errors.append(f"Missing key: {full_path}")
        elif isinstance(expected_type, dict):
            if not isinstance(section_data.get(key), dict):
                errors.append(f"Expected dict for {full_path}, got {type(section_data.get(key)).__name__}")
            else:
                errors.extend(validate_section(section_data[key], expected_type, full_path))
        else:
            if not isinstance(section_data.get(key), expected_type):
                errors.append(f"Invalid type for {full_path}: expected {expected_type.__name__}, got {type(section_data.get(key)).__name__}")
    return errors

def validate_agent_profile(profile_path: str) -> bool:
    with open(profile_path, "r") as f:
        profile = yaml.safe_load(f)

    errors = []

    for section, keys in REQUIRED_KEYS.items():
        if section not in profile:
            errors.append(f"Missing section: {section}")
        else:
            errors.extend(validate_section(profile[section], keys, section))

    if errors:
        print(f"[❌] Profile '{profile_path}' is invalid:")
        for e in errors:
            print(f"   - {e}")
        return False

    print(f"[✅] Profile '{profile_path}' is valid.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate YAML agent profiles.")
    parser.add_argument("--path", type=str, default="profiles/agents", help="Path to agent profiles directory")
    args = parser.parse_args()

    profiles_dir = Path(args.path)
    for profile_file in profiles_dir.glob("*.yaml"):
        validate_agent_profile(str(profile_file))