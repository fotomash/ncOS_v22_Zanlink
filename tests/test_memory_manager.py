import time
import unittest

from memory_manager import EnhancedMemoryManager


class TestEnhancedMemoryManager(unittest.TestCase):
    def test_store_memory_purges_expired(self):
        mm = EnhancedMemoryManager(ttl_seconds=1, default_window=5)
        mm.store_memory("ns", "first")
        time.sleep(0.1)
        mm.store_memory("ns", "second")
        time.sleep(0.6)
        mm.store_memory("ns", "third")
        time.sleep(0.6)
        mm.store_memory("ns", "fourth")

        entries = mm.get_context_window("ns", 10)
        self.assertEqual([e.data for e in entries], ["third", "fourth"])

    def test_get_context_window_most_recent(self):
        mm = EnhancedMemoryManager(ttl_seconds=60)
        for i in range(5):
            mm.store_memory("chat", f"item{i}")

        window = mm.get_context_window("chat", 3)
        self.assertEqual([e.data for e in window], ["item2", "item3", "item4"])


if __name__ == "__main__":
    unittest.main()
