from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class AppDefinition:
    app_id: str
    aliases: frozenset[str]
    executable_candidates: tuple[Path, ...]


class AppRegistry:
    def __init__(self, definitions: tuple[AppDefinition, ...]) -> None:
        self._definitions: dict[str, AppDefinition] = {}
        self._aliases: dict[str, str] = {}
        for definition in definitions:
            if not definition.app_id or definition.app_id in self._definitions:
                raise ValueError(f"duplicate app id: {definition.app_id}")
            self._definitions[definition.app_id] = definition
            for alias in definition.aliases | {definition.app_id}:
                normalized = normalize_app_alias(alias)
                existing = self._aliases.get(normalized)
                if existing is not None and existing != definition.app_id:
                    raise ValueError(f"duplicate app alias: {alias}")
                self._aliases[normalized] = definition.app_id

    def resolve_alias(self, text: str) -> str | None:
        compact = normalize_app_alias(text)
        matches = [
            (alias, app_id)
            for alias, app_id in self._aliases.items()
            if alias in compact
        ]
        return max(matches, default=("", None), key=lambda item: len(item[0]))[1]

    def executable_for(self, app_id: str) -> Path | None:
        definition = self._definitions.get(app_id)
        if definition is None:
            return None
        return next((path for path in definition.executable_candidates if path.is_file()), None)

    @property
    def aliases(self) -> frozenset[str]:
        return frozenset(self._aliases)


def normalize_app_alias(value: str) -> str:
    return "".join(value.lower().split())


def default_app_registry() -> AppRegistry:
    windows = Path(os.environ.get("WINDIR", r"C:\Windows"))
    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    local_app_data = Path(os.environ.get("LOCALAPPDATA", ""))
    candidates = (
        AppDefinition(
            "chrome",
            frozenset({"크롬", "chrome"}),
            (
                program_files / "Google/Chrome/Application/chrome.exe",
                local_app_data / "Google/Chrome/Application/chrome.exe",
            ),
        ),
        AppDefinition(
            "notepad",
            frozenset({"메모장", "notepad"}),
            (windows / "System32/notepad.exe",),
        ),
        AppDefinition(
            "calculator",
            frozenset({"계산기", "calculator", "calc"}),
            (windows / "System32/calc.exe",),
        ),
        AppDefinition(
            "vscode",
            frozenset({"vscode", "비주얼스튜디오코드", "비주얼 스튜디오 코드"}),
            (
                Path(r"C:\Microsoft VS Code\Code.exe"),
                local_app_data / "Programs/Microsoft VS Code/Code.exe",
                program_files / "Microsoft VS Code/Code.exe",
            ),
        ),
    )
    return AppRegistry(candidates)
