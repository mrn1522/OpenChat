import argparse
import os

import uvicorn

from app.config import reload_settings
from app.main import app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OpenChat API server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir")
    args = parser.parse_args()

    if args.data_dir:
        os.environ["OPENCHAT_DATA_DIR"] = args.data_dir
        reload_settings()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
