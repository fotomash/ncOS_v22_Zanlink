import types
import pytest
import sys

class DummyRequestsModule(types.ModuleType):
    class RequestException(Exception):
        pass

    def __init__(self):
        super().__init__('requests')
        self.post = lambda *a, **k: None

sys.modules['requests'] = DummyRequestsModule()
import requests

sys.modules.setdefault('dotenv', types.ModuleType('dotenv'))
sys.modules['dotenv'].load_dotenv = lambda *a, **k: None
sys.modules.setdefault('dotenv.main', types.ModuleType('main'))

import os
os.environ.setdefault('DX_PRIVATE_KEY', 'dummy')
os.environ.setdefault('DX_PUBLIC_KEY', 'dummy')
os.environ.setdefault('DX_ACCOUNT_CODE', 'dummy')

import src.ncos.engines.zdx_core as zdx


def test_place_order_network_failure(monkeypatch):
    def mock_post(*args, **kwargs):
        raise requests.RequestException("network down")
    monkeypatch.setattr(zdx.requests, "post", mock_post)
    result = zdx.place_order("EURUSD", 1.1)
    assert result is None
