import importlib.util
import sys
import types
import unittest


class TestVoiceTagParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # stub spacy so the parser can be imported without the dependency
        sys.modules['spacy'] = types.SimpleNamespace(load=lambda name: None)
        spec = importlib.util.spec_from_file_location('voice_tag_parser', 'ncOS/voice_tag_parser.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.parser = module.VoiceTagParser()
        cls.module = module

    def test_entity_extraction_from_docs_example(self):
        tag = self.parser.parse("mark gold bullish on H4")
        self.assertEqual(tag.symbol, "XAUUSD")
        self.assertEqual(tag.timeframe, "H4")
        self.assertEqual(tag.bias, "bullish")
        self.assertEqual(tag.action, "mark")

    def test_menu_action_generation(self):
        tag = self.parser.parse("mark gold bullish on H4")
        action = self.parser.to_menu_action(tag)
        self.assertEqual(action["action"], "append_journal")
        self.assertEqual(action["params"]["symbol"], "XAUUSD")
        self.assertEqual(action["params"]["timeframe"], "H4")


if __name__ == "__main__":
    unittest.main()
