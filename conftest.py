import sys
import os
import pytest

# Thêm project root vào sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Set up test environment variables."""
    os.environ["SUPABASE_URL"] = "http://test.supabase.co"
    os.environ["SUPABASE_SERVICE_KEY"] = "test_service_key"
    os.environ["GEMINI_API_KEY"] = "test_gemini_key"