"""Protection for secrets persisted in the settings file.

On Windows the API key is encrypted with DPAPI (``CryptProtectData``) before
being written to ``.env`` and stored as ``dpapi:<base64>``. DPAPI binds the
cipher to the current Windows user, so the file is unreadable by other users
or after being copied elsewhere. On other platforms values pass through
unchanged (``secrets_store`` is a no-op there and ``.env`` keeps its 0600
permissions).

On Windows the key is additionally mirrored into Windows Credential Manager
so it survives the app-data directory being recreated — reinstalls that wipe
``%APPDATA%\\com.openchat.desktop`` no longer force a settings round-trip.
Other platforms have no managed credential store here and are no-ops.
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


CREDENTIAL_TARGET = "OpenChat/openrouter-api-key"

_CRED_TYPE_GENERIC = 1
_CRED_PERSIST_LOCAL_MACHINE = 2
_ERROR_NOT_FOUND = 1168


class _Credential(ctypes.Structure):
    # CREDENTIALW from wincred.h; fields the code never sets stay zeroed.
    _fields_ = [
        ("Flags", ctypes.wintypes.DWORD),
        ("Type", ctypes.wintypes.DWORD),
        ("TargetName", ctypes.wintypes.LPWSTR),
        ("Comment", ctypes.wintypes.LPWSTR),
        ("LastWritten", ctypes.wintypes.FILETIME),
        ("CredentialBlobSize", ctypes.wintypes.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_char)),
        ("Persist", ctypes.wintypes.DWORD),
        ("AttributeCount", ctypes.wintypes.DWORD),
        ("Attributes", ctypes.c_void_p),
        ("TargetAlias", ctypes.wintypes.LPWSTR),
        ("UserName", ctypes.wintypes.LPWSTR),
    ]


def _advapi32() -> "ctypes.WinDLL":
    return ctypes.WinDLL("advapi32", use_last_error=True)


def _cred_read() -> str:
    advapi32 = _advapi32()
    cred_ptr = ctypes.POINTER(_Credential)()
    if not advapi32.CredReadW(
        CREDENTIAL_TARGET, _CRED_TYPE_GENERIC, 0, ctypes.byref(cred_ptr)
    ):
        if ctypes.get_last_error() == _ERROR_NOT_FOUND:
            return ""
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        blob = cred_ptr.contents.CredentialBlob
        return ctypes.string_at(blob, cred_ptr.contents.CredentialBlobSize).decode(
            "utf-8"
        )
    finally:
        advapi32.CredFree(cred_ptr)


def _cred_write(value: str) -> None:
    blob = ctypes.create_string_buffer(value.encode("utf-8"))
    cred = _Credential(
        Type=_CRED_TYPE_GENERIC,
        TargetName=CREDENTIAL_TARGET,
        # Exclude create_string_buffer's trailing NUL from the stored blob.
        CredentialBlobSize=len(blob.raw) - 1,
        CredentialBlob=ctypes.cast(blob, ctypes.POINTER(ctypes.c_char)),
        Persist=_CRED_PERSIST_LOCAL_MACHINE,
        UserName="openchat",
    )
    if not _advapi32().CredWriteW(ctypes.byref(cred), 0):
        raise ctypes.WinError(ctypes.get_last_error())


def _cred_delete() -> None:
    if not _advapi32().CredDeleteW(CREDENTIAL_TARGET, _CRED_TYPE_GENERIC, 0):
        if ctypes.get_last_error() == _ERROR_NOT_FOUND:
            return
        raise ctypes.WinError(ctypes.get_last_error())


def read_credential() -> str:
    """Return the API key mirrored in the OS credential store, or ""."""
    if os.name != "nt":
        return ""
    try:
        return _cred_read()
    except Exception:
        logger.exception("Failed to read the stored credential")
        return ""


def write_credential(value: str) -> None:
    """Mirror ``value`` into the OS credential store (best effort)."""
    if not value or os.name != "nt":
        return
    try:
        _cred_write(value)
    except Exception:
        logger.exception("Failed to write the stored credential")


def delete_credential() -> None:
    """Drop the mirrored credential (best effort)."""
    if os.name != "nt":
        return
    try:
        _cred_delete()
    except Exception:
        logger.exception("Failed to delete the stored credential")
