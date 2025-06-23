import sys
import types
import unittest

# Provide minimal yaml stub if PyYAML is not installed
if 'yaml' not in sys.modules:
    yaml_stub = types.ModuleType('yaml')
    yaml_stub.dump = lambda data, sort_keys=False: 'yaml'
    sys.modules['yaml'] = yaml_stub

# Import from integrated package
from menu_system import EnhancedMenuSystem


class DummyOrchestrator:
    def generate_enhanced_menu(self):
        return {"title": "Test Menu", "categories": {}}

    agents = {}


class TestVoiceMenuSystem(unittest.TestCase):
    def test_instantiation(self):
        menu = EnhancedMenuSystem(DummyOrchestrator())
        self.assertEqual(menu.generate_menu()["title"], "Test Menu")


if __name__ == "__main__":
    unittest.main()
