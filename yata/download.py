from __future__ import annotations

import errno
import hashlib
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .catalog import ModelEntry, RemoteFile
from .hardware import disk_free
from .i18n import t

CHUNK = 1024 * 1024
PARTIAL = ".yata-download"
MARGIN = 256 * 1024 * 1024
USER_AGENT = "Yata-model-setup/1"

Progress = Callable[[int, int, str, str], None]


class DownloadError(RuntimeError):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def format_bytes(value: int | float | None) -> str:
    if value is None:
        return t("不明", "unknown")
    for unit, size in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
        if value >= size:
            return f"{value / size:.1f} {unit}"
    return f"{int(value)} B"


def inspect(entry: ModelEntry, folder: Path) -> dict[str, Any]:
    present, missing, mismatched = [], [], []
    for item in entry.files:
        target = folder / item.path
        try:
            size = target.stat().st_size if target.is_file() else None
        except OSError:
            size = None
        if size is None:
            missing.append(item.path)
        elif size == item.size:
            present.append(item.path)
        else:
            mismatched.append(item.path)
    remaining = sum(item.size for item in entry.files if item.path in missing)
    usable = all(name in present for name in entry.essential) and not mismatched
    return {"present": present, "missing": missing, "mismatched": mismatched, "remaining": remaining, "usable": usable,
            "complete": not missing and not mismatched}


def fingerprint(entry: ModelEntry, folder: Path) -> dict[str, list[int]]:
    result = {}
    for item in entry.files:
        try:
            stat = (folder / item.path).stat()
        except OSError:
            continue
        result[item.path] = [stat.st_size, stat.st_mtime_ns]
    return result


def _digest(path: Path, item: RemoteFile, on_bytes: Callable[[int], None], cancelled: Callable[[], bool]) -> bool | None:
    if item.sha256:
        hasher = hashlib.sha256()
        expected = item.sha256
    elif item.git_sha1:
        hasher = hashlib.sha1(usedforsecurity=False)
        hasher.update(b"blob %d\0" % path.stat().st_size)
        expected = item.git_sha1
    else:
        return None
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            if cancelled():
                raise DownloadError("cancelled", t("中止しました。", "Cancelled."))
            hasher.update(chunk)
            on_bytes(len(chunk))
    return hasher.hexdigest() == expected


def verify_hashes(entry: ModelEntry, folder: Path, progress: Progress | None = None,
                  cancelled: Callable[[], bool] = lambda: False) -> list[str]:
    names = inspect(entry, folder)["present"]
    total = sum(item.size for item in entry.files if item.path in names)
    done = 0
    differing = []
    for item in entry.files:
        if item.path not in names:
            continue

        def advance(count: int, name: str = item.path) -> None:
            nonlocal done
            done += count
            if progress:
                progress(done, total, name, "verify")

        if _digest(folder / item.path, item, advance, cancelled) is False:
            differing.append(item.path)
    return differing


def _classify(exc: BaseException) -> DownloadError:
    if isinstance(exc, DownloadError):
        return exc
    if isinstance(exc, urllib.error.HTTPError):
        return DownloadError("network", t(f"配布元がエラーを返しました（HTTP {exc.code}）。時間をおいて再試行してください。",
                                          f"The source returned an error (HTTP {exc.code}). Try again later."))
    if isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError)):
        reason = getattr(exc, "reason", exc)
        return DownloadError("network", t(f"配布元に接続できません（{type(reason).__name__}）。ネットワークを確認して再試行してください。",
                                          f"Cannot reach the source ({type(reason).__name__}). Check the network and try again."))
    if isinstance(exc, PermissionError):
        return DownloadError("permission", t(f"保存先に書き込めません（{exc.filename or ''}）。別の保存先を選んでください。",
                                             f"Cannot write to the folder ({exc.filename or ''}). Choose another folder."))
    if isinstance(exc, OSError) and (exc.errno == errno.ENOSPC or getattr(exc, "winerror", None) in (39, 112)):
        return DownloadError("disk", t("保存先の空き容量がなくなりました。空きを作るか別の保存先を選び、再試行してください。",
                                       "The folder ran out of space. Free some space or choose another folder, then try again."))
    if isinstance(exc, OSError):
        return DownloadError("disk", t(f"保存先への書き込みに失敗しました（{exc.strerror or type(exc).__name__}）。",
                                       f"Writing to the folder failed ({exc.strerror or type(exc).__name__})."))
    return DownloadError("network", t(f"取得に失敗しました（{type(exc).__name__}）。", f"The download failed ({type(exc).__name__})."))


def fetch(entry: ModelEntry, folder: Path, *, progress: Progress | None = None, cancelled: Callable[[], bool] = lambda: False,
          opener: Callable[..., Any] = urllib.request.urlopen, margin: int = MARGIN) -> None:
    try:
        _fetch(entry, folder, progress, cancelled, opener, margin)
    except Exception as exc:  # noqa: BLE001 - every failure leaves as a classified DownloadError
        raise _classify(exc) from exc


def _fetch(entry: ModelEntry, folder: Path, progress: Progress | None, cancelled: Callable[[], bool],
           opener: Callable[..., Any], margin: int) -> None:
    state = inspect(entry, folder)
    if state["mismatched"]:
        names = "、".join(state["mismatched"])
        raise DownloadError("conflict", t(f"{names} が配布元と異なるサイズで既にあります。上書きしないため、移動するか別の保存先を選んでください。",
                                          f"{names} already exist with sizes that differ from the source. Yata does not overwrite them: "
                                          "move them or choose another folder."))
    partial = folder / PARTIAL
    parts = {item.path: partial / (item.path + ".part") for item in entry.files if item.path in state["missing"]}
    already = {name: (path.stat().st_size if path.is_file() else 0) for name, path in parts.items()}
    needed = state["remaining"] - sum(already.values())
    free = disk_free(folder)
    if free is not None and free < needed + margin:
        raise DownloadError("disk", t(f"空き容量が足りません（必要 {format_bytes(needed + margin)}、空き {format_bytes(free)}）。"
                                      "空きを作るか、別の保存先を選んでください。",
                                      f"Not enough free space (needs {format_bytes(needed + margin)}, free {format_bytes(free)}). "
                                      "Free some space or choose another folder."))
    total = entry.size
    done = sum(item.size for item in entry.files if item.path in state["present"]) + sum(already.values())
    if progress:
        progress(done, total, "", "download")
    if parts:
        partial.mkdir(parents=True, exist_ok=True)
    for item in entry.files:
        if item.path not in parts:
            continue
        part = parts[item.path]
        offset = part.stat().st_size if part.is_file() else 0
        if offset > item.size:
            part.unlink()
            done -= offset
            offset = 0
        if offset < item.size:
            headers = {"User-Agent": USER_AGENT}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            try:
                response = opener(urllib.request.Request(entry.url(item), headers=headers), timeout=30)
            except urllib.error.HTTPError as exc:
                if exc.code != 416:
                    raise
                # The saved part no longer matches what the source serves: start the file again.
                part.unlink(missing_ok=True)
                done -= offset
                offset = 0
                response = opener(urllib.request.Request(entry.url(item), headers={"User-Agent": USER_AGENT}), timeout=30)
            with response:
                if offset and getattr(response, "status", 200) != 206:
                    done -= offset  # the source ignored the range: the whole file comes again
                    offset = 0
                with part.open("ab" if offset else "wb") as out:
                    written = offset
                    while chunk := response.read(CHUNK):
                        if cancelled():
                            raise DownloadError("cancelled", t("中止しました。途中まで取得した分は次回に続きから使います。",
                                                               "Cancelled. The part already fetched is used next time."))
                        written += len(chunk)
                        if written > item.size:
                            break
                        out.write(chunk)
                        done += len(chunk)
                        if progress:
                            progress(done, total, item.path, "download")
        size = part.stat().st_size
        if size != item.size:
            part.unlink(missing_ok=True)
            raise DownloadError("corrupt", t(f"{item.path} のサイズが配布元と一致しません（{size} / {item.size} バイト）。再試行してください。",
                                         f"{item.path} does not match the source's size ({size} / {item.size} bytes). Try again."))
        if progress:
            progress(done, total, item.path, "verify")
        if _digest(part, item, lambda _count: None, cancelled) is False:
            part.unlink(missing_ok=True)
            raise DownloadError("corrupt", t(f"{item.path} が壊れていました（ハッシュが配布元と一致しません）。再試行してください。",
                                         f"{item.path} was damaged (its hash does not match the source). Try again."))
        target = folder / item.path
        if target.exists():
            raise DownloadError("conflict", t(f"{item.path} が取得中に別の手段で置かれました。上書きしないため中断します。",
                                          f"{item.path} was put there by something else during the download. Yata stops rather than overwrite it."))
        os.replace(part, target)
    if partial.is_dir() and not any(partial.iterdir()):
        shutil.rmtree(partial, ignore_errors=True)
