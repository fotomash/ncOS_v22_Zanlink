# Fix for test imports
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Create mock zanflow module for tests
class MockZanflow:
    """Mock zanflow module to prevent import errors"""

    class Bridge:
        def __init__(self, *args, **kwargs):
            pass

        def connect(self):
            return True

        def analyze(self, data):
            return {"status": "mocked", "result": "test"}

    class DataManager:
        def __init__(self, *args, **kwargs):
            pass

        def load(self, symbol):
            return {"symbol": symbol, "data": []}

# Create zanflow mock
sys.modules['zanflow'] = MockZanflow()

# Run this before importing any test modules
if __name__ == "__main__":
    print("✅ Test environment configured")
    print("   - Added project root to Python path")
    print("   - Created mock zanflow module")
    print("   - Ready to run tests")
