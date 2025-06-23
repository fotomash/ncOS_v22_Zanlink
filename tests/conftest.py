import sys
from unittest.mock import MagicMock


def pytest_configure(config):
    # Provide lightweight stubs for optional heavy dependencies
    stubs = [
        "spacy",
        "yaml",
        "requests",
        "zbar_agent",
        "zbar_logger",
        "engine",
    ]
    for name in stubs:
        sys.modules.setdefault(name, MagicMock())
