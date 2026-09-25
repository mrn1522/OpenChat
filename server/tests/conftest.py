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
os.environ.pop("OPENAI_API_KEY", None)

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True, scope="session")
def _isolate_credential_store():
    import app.config as config
    import app.main as main_module

    # A separate patch context so a test-level monkeypatch.undo() restores
    # these stubs rather than the real helpers — on Windows the real ones
    # would overwrite/delete the developer's actual saved credential.
    with pytest.MonkeyPatch.context() as isolated:
        isolated.setattr(config, "read_credential", lambda: "")
        isolated.setattr(main_module, "read_credential", lambda: "")
        isolated.setattr(main_module, "write_credential", lambda value: True)
        isolated.setattr(main_module, "delete_credential", lambda: True)
        yield


@pytest.fixture(scope="session")
def client():
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
