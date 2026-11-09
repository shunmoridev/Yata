from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser

from . import PORT, i18n
from .archive import default_data_dir
from .i18n import t

URL = f"http://127.0.0.1:{PORT}/"


def port_state() -> str:
    try:
        with socket.create_connection(("127.0.0.1", PORT), timeout=2):
            pass
    except OSError:
        return "free"
    # No proxy: the page is on this PC.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(URL, timeout=5) as response:
            page = response.read(65_536).decode("utf-8", "replace")
    except (OSError, ValueError):
        return "other"
    # The page names itself in <meta name="application-name">; versions before that had "YATA" in the title.
    return "yata" if 'name="application-name" content="Yata"' in page or "<title>YATA" in page else "other"


def _open_when_ready(timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if port_state() == "yata":
            webbrowser.open(URL)
            return
        time.sleep(0.5)


def _saved_locale() -> str | None:
    try:
        data = json.loads((default_data_dir() / "setup.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return i18n.normalize(data.get("locale")) if isinstance(data, dict) else None


def main(argv: list[str] | None = None) -> int:
    # Piped or redirected output uses the console code page (cp932 on Japanese Windows), which cannot
    # encode every character in these lines; replace those rather than fail at startup.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    i18n.set_choice(_saved_locale())
    parser = argparse.ArgumentParser(prog="python -m yata", description=t("Yata をこの PC（127.0.0.1）で起動します。",
                                                                          "Start Yata on this PC (127.0.0.1)."))
    parser.add_argument("--open", action="store_true", help=t("起動したら、既定のブラウザで Yata のページを開く",
                                                              "open Yata's page in the default browser once it runs"))
    args = parser.parse_args(argv)

    state = port_state()
    if state == "yata":
        print(t(f"Yata はもう起動しています: {URL}", f"Yata is already running: {URL}"))
        if args.open:
            webbrowser.open(URL)
        return 0
    if state == "other":
        print(t(f"ポート {PORT} を別のアプリが使っているため、Yata を起動できません。"
                "そのアプリを終了してから、もう一度起動してください。",
                f"Another app is using port {PORT}, so Yata cannot start. Quit that app and start Yata again."),
              file=sys.stderr)
        return 1

    import uvicorn

    from .api import app  # built only now, once the port is known to be free

    print(f"Yata local UI: {URL}")
    settings = app.state.asr.settings
    if settings is not None:
        print("ASR:", settings.describe())
    print(t("終了するにはこのウィンドウで Ctrl+C を押してください。", "To stop Yata, press Ctrl+C in this window."))
    if args.open:
        threading.Thread(target=_open_when_ready, name="yata-open-browser", daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT, access_log=False, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
