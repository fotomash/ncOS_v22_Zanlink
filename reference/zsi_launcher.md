# ZSI Launcher

```python
# ZSI Launcher - Pro CTO-approved
import yaml
import json
import os
import argparse
import logging
from dotenv import load_dotenv
import pkgutil
import importlib

# --- Logging Configuration ---
logger = logging.getLogger("zsi_launcher")
logger.setLevel(os.getenv("ZSI_LOG_LEVEL", "INFO"))
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
logger.addHandler(ch)

def discover_agents(modules_dir):
    registry = {}
    pkg_path = os.path.abspath(modules_dir)
    for finder, name, ispkg in pkgutil.iter_modules([pkg_path]):
        module_name = f"{modules_dir.replace(os.sep,'.')}.{name}.agent"
        try:
            mod = importlib.import_module(module_name)
            handler = getattr(mod, "handle_intent", None)
            if callable(handler):
                registry[name] = handler
                logger.info(f"Registered agent: {name}")
        except Exception as err:
            logger.warning(f"Skipping agent {name}: {err}")
    return registry

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="ZSI Copilot Framework Launcher")
    parser.add_argument("-c", "--config", default=os.getenv("ZSI_CONFIG_PATH","zsi_config.yaml"), help="Path to config YAML")
    parser.add_argument("-b", "--blueprint", default=os.getenv("ZSI_BLUEPRINT_PATH","startup_blueprint.json"), help="Path to blueprint JSON")
    parser.add_argument("-m", "--modules", dest="modules_path", default=os.getenv("ZSI_MODULES_PATH","modules"), help="Directory for agent modules")
    parser.add_argument("--mission", default=os.getenv("ZSI_MISSION","Unknown Mission"), help="Override mission description")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel("DEBUG")
        logger.debug("Verbose logging enabled")

    # Load config
    try:
        with open(args.config) as cf:
            config = yaml.safe_load(cf)
        logger.info(f"Loaded config from {args.config}")
    except Exception as e:
        logger.critical(f"Failed to load config: {e}", exc_info=True)
        return 1

    # Load blueprint
    try:
        with open(args.blueprint) as bf:
            blueprint = json.load(bf)
        logger.info(f"Loaded blueprint from {args.blueprint}")
    except Exception as e:
        logger.critical(f"Failed to load blueprint: {e}", exc_info=True)
        return 1

    # Discover agents
    agents = discover_agents(args.modules_path)

    # Launch summary
    logger.info("Launching ZSI Agent Framework")
    logger.info(f"Mission: {args.mission or config.get('mission')}")
    logger.info(f"Blueprint: {blueprint.get('name')}")
    logger.info(f"Agents: {', '.join(agents.keys())}")

if __name__ == "__main__":
    exit(main())
```