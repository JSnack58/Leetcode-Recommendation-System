import json
from pathlib import Path


class JsonlWriter:
    """Append-mode JSONL writer with built-in resume tracking.

    Usage:
        with JsonlWriter(path) as w:
            if not w.is_done(page):
                w.write_page(page, entries)
    """

    def __init__(self, output_file: Path) -> None:
        self.output_file = output_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        self._completed: set[int] = self._load_completed_pages()
        self._fh = open(output_file, "a", encoding="utf-8")

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *args) -> None:
        self._fh.close()

    def is_done(self, page: int) -> bool:
        return page in self._completed

    def write_page(self, page: int, entries: list[dict]) -> None:
        if entries:
            for entry in entries:
                entry["_page"] = page
                self._fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        else:
            # Sentinel: marks the page complete so it isn't retried on resume
            self._fh.write(json.dumps({"_page": page, "_empty": True}) + "\n")
        self._fh.flush()
        self._completed.add(page)

    def reset(self) -> None:
        """Delete the output file and start fresh."""
        self._fh.close()
        self.output_file.unlink(missing_ok=True)
        self._completed = set()
        self._fh = open(self.output_file, "a", encoding="utf-8")

    def _load_completed_pages(self) -> set[int]:
        completed: set[int] = set()
        if not self.output_file.exists():
            return completed
        with open(self.output_file, encoding="utf-8") as f:
            for line in f:
                try:
                    completed.add(json.loads(line)["_page"])
                except (json.JSONDecodeError, KeyError):
                    pass
        return completed
