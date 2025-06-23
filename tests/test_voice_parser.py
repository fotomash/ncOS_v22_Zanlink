import sys
import unittest
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _mock_deps(monkeypatch):
    monkeypatch.setitem(sys.modules, "spacy", MagicMock())
    monkeypatch.setitem(sys.modules, "yaml", MagicMock())
    monkeypatch.setitem(sys.modules, "zbar_agent", MagicMock())
    monkeypatch.setitem(sys.modules, "requests", MagicMock())
    monkeypatch.setitem(sys.modules, "zbar_logger", MagicMock())
    monkeypatch.setitem(sys.modules, "engine", MagicMock())


from ncOS import voice_tag_parser


class TestVoiceTagParser(unittest.TestCase):
    def setUp(self):
        self.parser = voice_tag_parser.VoiceTagParser()

    def test_mark_phrase_parsing(self):
        tag = self.parser.parse('Mark gold bullish on 4hour swept lows at 2358')
        self.assertEqual(tag.symbol, 'XAUUSD')
        self.assertEqual(tag.timeframe, 'H4')
        self.assertEqual(tag.bias, 'bullish')
        self.assertEqual(tag.notes, 'swept lows 2358')

    def test_log_phrase_parsing(self):
        tag = self.parser.parse('Log euro bearish 1min after NFP')
        self.assertEqual(tag.symbol, 'EURUSD')
        self.assertEqual(tag.timeframe, 'M1')
        self.assertEqual(tag.bias, 'bearish')
        self.assertEqual(tag.notes, 'after nfp')


if __name__ == '__main__':
    unittest.main()
