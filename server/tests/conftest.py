import os
import tempfile

# Isolate persistence: settings resolve the history DB and .env from the data
# dir at import time, so these must be force-set (not setdefault) before app
# modules are imported — a developer shell may already export real paths.
_test_data_dir = tempfile.TemporaryDirectory(prefix="openchat-test-")
os.environ["OPENCHAT_DATA_DIR"] = _test_data_dir.name
os.environ["OPENCHAT_HISTORY_DB_PATH"] = os.path.join(
    _test_data_dir.name, "openchat_history.db"
)

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client():
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
