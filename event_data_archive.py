"""Event data helpers, shared by the Add-On and the archive refresh script."""

import json
from pathlib import Path

ARCHIVE_PATH = Path(__file__).parent / "event_data" / "archive.json"


def load_archive(path=ARCHIVE_PATH):
    """Loads the archived event data, or {} if there is no archive."""

    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_archive(event_data, path=ARCHIVE_PATH):
    """Writes the archived event data sorted by project name, one project per line."""

    path.parent.mkdir(exist_ok=True)

    lines = [
        f"{json.dumps(project_name, ensure_ascii=False)}: "
        f"{json.dumps(event_data[project_name], ensure_ascii=False)}"
        for project_name in sorted(event_data)
    ]

    with open(path, "w", encoding="utf-8") as file:
        file.write("{\n" + ",\n".join(lines) + "\n}\n")
