"""Protection for secrets persisted in the settings file.

On Windows the API key is encrypted with DPAPI (``CryptProtectData``) before
being written to ``.env`` and stored as ``dpapi:<base64>``. DPAPI binds the
cipher to the current Windows user, so the file is unreadable by other users
or after being copied elsewhere. On other platforms values pass through
unchanged (``secrets_store`` is a no-op there and ``.env`` keeps its 0600
permissions).
"""

import base64
import ctypes
import ctypes.wintypes
import logging
import os

logger = logging.getLogger(__name__)

DPAPI_PREFIX = "dpapi:"


def _crypt32() -> "ctypes.WinDLL":
    # use_last_error is required or get_last_error() returns ctypes' private
    # (always-zero) copy instead of the real Windows error code.
    return ctypes.WinDLL("crypt32", use_last_error=True)


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", ctypes.wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_char)),
    ]


def _to_blob(data: bytes) -> tuple[_DataBlob, ctypes.Array]:
    buffer = ctypes.create_string_buffer(data, len(data))
    return (
        _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char))),
        buffer,
    )


def _crypt_protect(data: bytes) -> bytes:
    blob_in, _keep = _to_blob(data)
    blob_out = _DataBlob()
    crypt32 = _crypt32()
    kernel32 = ctypes.windll.kernel32
    if not crypt32.CryptProtectData(
        ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        kernel32.LocalFree(blob_out.pbData)


def _crypt_unprotect(data: bytes) -> bytes:
    blob_in, _keep = _to_blob(data)
    blob_out = _DataBlob()
    crypt32 = _crypt32()
    kernel32 = ctypes.windll.kernel32
    if not crypt32.CryptUnprotectData(
        ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        kernel32.LocalFree(blob_out.pbData)


def protect_secret(value: str) -> str:
    """Return the at-rest form of ``value`` (DPAPI-encrypted on Windows)."""
    if not value or os.name != "nt":
        return value
    protected = _crypt_protect(value.encode("utf-8"))
    return DPAPI_PREFIX + base64.b64encode(protected).decode("ascii")


def unprotect_secret(value: str) -> str:
    """Decode a stored secret value, returning "" if it cannot be recovered."""
    if not value.startswith(DPAPI_PREFIX):
        return value
    if os.name != "nt":
        logger.warning("Skipping DPAPI-protected secret on a non-Windows host")
        return ""
    try:
        payload = base64.b64decode(value[len(DPAPI_PREFIX):])
        return _crypt_unprotect(payload).decode("utf-8")
    except Exception:
        logger.exception("Failed to decrypt stored secret; treating it as unset")
        return ""
