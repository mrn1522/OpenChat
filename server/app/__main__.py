import argparse
import os
import threading
import time

import uvicorn


def _parent_is_alive(parent_pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        synchronize = 0x00100000
        wait_timeout = 0x102
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(synchronize, False, parent_pid)
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == wait_timeout
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(parent_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _start_parent_watchdog(parent_pid: int) -> None:
    def monitor() -> None:
        while True:
            time.sleep(1)
            if not _parent_is_alive(parent_pid):
                os._exit(0)

    threading.Thread(target=monitor, name="parent-watchdog", daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OpenChat API server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir")
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args()

    if args.data_dir:
        os.environ["OPENCHAT_DATA_DIR"] = args.data_dir

    from app.main import app

    if args.parent_pid is not None:
        _start_parent_watchdog(args.parent_pid)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
