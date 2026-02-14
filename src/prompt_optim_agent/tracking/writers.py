"""Local file writers for experiment metrics."""

import csv
import json
import os


class LocalStore:
    """Writes metrics, tables, and config to local files."""

    def __init__(self, log_dir: str | None = None):
        self.log_dir = log_dir

    def append_metrics(self, metrics: dict) -> None:
        if not self.log_dir:
            return
        path = os.path.join(self.log_dir, "metrics.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(_make_serializable(metrics)) + "\n")

    def write_csv(self, name: str, columns: list, data: list) -> None:
        if not self.log_dir:
            return
        tables_dir = os.path.join(self.log_dir, "tables")
        os.makedirs(tables_dir, exist_ok=True)
        path = os.path.join(tables_dir, f"{name}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(data)

    def write_json(self, filename: str, data: dict) -> None:
        if not self.log_dir:
            return
        path = os.path.join(self.log_dir, filename)
        with open(path, "w") as f:
            json.dump(_make_serializable(data), f, indent=2)


def _make_serializable(data: dict) -> dict:
    """Return a copy of *data* where non-serializable values are stringified."""
    out = {}
    for key, value in data.items():
        try:
            json.dumps(value)
            out[key] = value
        except (TypeError, ValueError):
            out[key] = str(value)
    return out
