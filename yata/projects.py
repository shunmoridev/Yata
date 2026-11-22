from __future__ import annotations

import json
import re
import threading
import uuid
from pathlib import Path
from typing import Any

from .archive import _replace
from .i18n import t
from .models import utc_now

FORMAT = 1
MAX_PROJECTS = 200
MAX_NAME_CHARS = 80
PROJECT_ID = re.compile(r"[0-9a-f]{32}")


class ProjectError(ValueError):
    pass


class ProjectStore:
    """Named groups for sessions. A session holds its project's id; deleting a project leaves its sessions unassigned."""

    def __init__(self, root: str | Path | None = None) -> None:
        # Without a root the projects live in memory only (tests, a session store with no archive).
        self.path = Path(root) / "projects.json" if root is not None else None
        self._lock = threading.RLock()
        self._projects: list[dict[str, Any]] | None = None

    def _load(self) -> list[dict[str, Any]]:
        if self._projects is not None:
            return self._projects
        projects: list[dict[str, Any]] = []
        try:
            record = json.loads(self.path.read_text(encoding="utf-8")) if self.path is not None else {}
        except (OSError, ValueError):
            record = {}
        if isinstance(record, dict) and record.get("format") == FORMAT and isinstance(record.get("projects"), list):
            for item in record["projects"]:
                if isinstance(item, dict) and PROJECT_ID.fullmatch(str(item.get("id", ""))) and isinstance(item.get("name"), str):
                    projects.append({"id": item["id"], "name": item["name"][:MAX_NAME_CHARS],
                                     "created_at_utc": str(item.get("created_at_utc") or "")})
        self._projects = projects
        return projects

    def _write(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _replace(self.path, json.dumps({"format": FORMAT, "projects": self._projects}, ensure_ascii=False))

    @staticmethod
    def _name(value: Any) -> str:
        name = " ".join(value.split()) if isinstance(value, str) else ""
        if not name:
            raise ProjectError(t("プロジェクト名を入れてください。", "Enter a project name."))
        if len(name) > MAX_NAME_CHARS:
            raise ProjectError(t(f"プロジェクト名は {MAX_NAME_CHARS} 文字までです。", f"A project name can be up to {MAX_NAME_CHARS} characters."))
        return name

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._load()]

    def exists(self, project_id: str) -> bool:
        with self._lock:
            return any(item["id"] == project_id for item in self._load())

    def create(self, name: Any) -> dict[str, Any]:
        clean = self._name(name)
        with self._lock:
            projects = self._load()
            if len(projects) >= MAX_PROJECTS:
                raise ProjectError(t(f"プロジェクトは {MAX_PROJECTS} 個までです。", f"There can be up to {MAX_PROJECTS} projects."))
            project = {"id": uuid.uuid4().hex, "name": clean, "created_at_utc": utc_now()}
            projects.append(project)
            try:
                self._write()
            except OSError:
                projects.remove(project)
                raise
            return dict(project)

    def rename(self, project_id: str, name: Any) -> dict[str, Any]:
        clean = self._name(name)
        with self._lock:
            project = self._find(project_id)
            previous = project["name"]
            project["name"] = clean
            try:
                self._write()
            except OSError:
                project["name"] = previous
                raise
            return dict(project)

    def delete(self, project_id: str) -> None:
        with self._lock:
            project = self._find(project_id)
            projects = self._load()
            index = projects.index(project)
            projects.pop(index)
            try:
                self._write()
            except OSError:
                projects.insert(index, project)
                raise

    def _find(self, project_id: str) -> dict[str, Any]:
        project = next((item for item in self._load() if item["id"] == project_id), None)
        if project is None:
            raise ProjectError(t("プロジェクトが見つかりません。", "Project not found."))
        return project
