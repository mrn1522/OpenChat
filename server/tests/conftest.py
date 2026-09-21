import os
import tempfile

# Isolate persistence: settings resolve the history DB and .env from the data
# dir at import time, so this must be set before app modules are imported.
os.environ.setdefault("OPENCHAT_DATA_DIR", tempfile.mkdtemp(prefix="openchat-test-"))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client():
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
