from __future__ import annotations

import os
from dataclasses import dataclass, fields

_ON = {"1", "on", "true", "yes"}


@dataclass(frozen=True)
class Features:
    theme_switch: bool = False
    slack: bool = False

    def enabled(self) -> list[str]:
        return [item.name for item in fields(self) if getattr(self, item.name)]

    def public(self) -> dict[str, bool]:
        return {item.name: bool(getattr(self, item.name)) for item in fields(self)}


def env_name(feature: str) -> str:
    return "YATA_FEATURE_" + feature.upper()


def load_features(environ: dict[str, str] | None = None) -> Features:
    env = os.environ if environ is None else environ
    return Features(**{item.name: env.get(env_name(item.name), "").strip().lower() in _ON for item in fields(Features)})
