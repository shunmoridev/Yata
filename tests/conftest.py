import os
import shutil
import socket
import tempfile
import traceback

import pytest

_DATA_DIR = tempfile.mkdtemp(prefix="yata-test-")
_HF_CACHE = tempfile.mkdtemp(prefix="yata-test-hf-")
_OUTSIDE: list[tuple[str, str]] = []


class OutsideNetwork(OSError):
    pass


def _loopback(host) -> bool:
    name = str(host or "").strip("[]").lower()
    return name in {"", "localhost", "::1"} or name.startswith("127.") or name.endswith(".localhost")


def _refuse(host, what: str) -> None:
    if _loopback(host):
        return
    # The innermost frames outside this file say which code tried it.
    origin = " <- ".join(f"{os.path.basename(frame.filename)}:{frame.lineno}" for frame in traceback.extract_stack()[-6:-2][::-1])
    _OUTSIDE.append((f"{what} {host}", origin))
    raise OutsideNetwork(f"tests stay on this PC: {what} {host} refused ({origin})")


def _guard_network() -> None:
    real_getaddrinfo = socket.getaddrinfo
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def getaddrinfo(host, *args, **kwargs):
        _refuse(host, "name lookup")
        return real_getaddrinfo(host, *args, **kwargs)

    def connect(self, address):
        if isinstance(address, tuple):
            _refuse(address[0], "connection to")
        return real_connect(self, address)

    def connect_ex(self, address):
        if isinstance(address, tuple):
            _refuse(address[0], "connection to")
        return real_connect_ex(self, address)

    socket.getaddrinfo = getaddrinfo
    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex


def pytest_configure(config):
    # Runs before test modules are imported, so yata.api's module-level app picks it up too.
    os.environ["YATA_DATA_DIR"] = _DATA_DIR
    os.environ["HF_HUB_CACHE"] = _HF_CACHE
    # The module-level app must not find the user's model folder either.
    os.environ.pop("YATA_MODEL_ROOT", None)
    _guard_network()


@pytest.fixture(autouse=True)
def _japanese_interface(monkeypatch):
    from yata import i18n

    monkeypatch.setattr(i18n, "system_locale", lambda: None)
    i18n.reset()
    yield
    i18n.reset()


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if _OUTSIDE:
        terminalreporter.section("connections outside this PC (refused)")
        for target, origin in _OUTSIDE:
            terminalreporter.line(f"{target}  from {origin}")


def pytest_sessionfinish(session, exitstatus):
    # A refused attempt fails the run even when the code under test swallowed the error.
    if _OUTSIDE and session.exitstatus == 0:
        session.exitstatus = 1


def pytest_unconfigure(config):
    shutil.rmtree(_DATA_DIR, ignore_errors=True)
    shutil.rmtree(_HF_CACHE, ignore_errors=True)
