"""Experiment tracking abstraction.

Wraps wandb so that all other modules can log metrics without importing wandb
directly. When wandb is disabled, metrics are stored locally as JSONL, CSV, and
JSON files so nothing is lost.
"""

import logging
from typing import Any

from .writers import LocalStore

logger = logging.getLogger(__name__)


class MetricsTracker:
    def __init__(
        self,
        wandb_config: dict | None = None,
        log_dir: str | None = None,
    ):
        self._enabled = wandb_config is not None
        self._run = None
        self._wandb = None
        self._local = LocalStore(log_dir)
        self._summary: dict[str, Any] = {}
        self._config: dict[str, Any] = {}

        if self._enabled:
            import wandb

            self._wandb = wandb
            wandb.init(config={}, **wandb_config)
            self._run = wandb.run

    def set_log_dir(self, log_dir: str) -> None:
        self._local.log_dir = log_dir

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------
    def log(self, metrics: dict) -> None:
        if self._enabled:
            self._wandb.log(metrics)
        self._local.append_metrics(metrics)

    def log_table(self, name: str, columns: list, data: list) -> None:
        if self._enabled:
            table = self._wandb.Table(columns=columns, data=data)
            self._wandb.log({name: table})
        self._local.write_csv(name, columns, data)

    def log_plot_table(self, name: str, vega_spec_name: str, data_table: Any, fields: dict) -> None:
        if self._enabled:
            tree = self._wandb.plot_table(
                vega_spec_name=vega_spec_name,
                data_table=data_table,
                fields=fields,
            )
            self._wandb.log({name: tree})
        if isinstance(data_table, dict):
            self._local.write_csv(
                name, data_table.get("columns", []), data_table.get("data", [])
            )

    def create_table(self, columns: list, data: list) -> Any:
        if self._enabled:
            return self._wandb.Table(columns=columns, data=data)
        return {"columns": columns, "data": data}

    def set_summary(self, key: str, value: Any) -> None:
        if self._enabled and self._run is not None:
            self._run.summary[key] = value
        self._summary[key] = value

    def set_config(self, config: dict) -> None:
        if self._enabled and self._run is not None:
            self._run.config.update(config)
        self._config.update(config)
        self._local.write_json("config.json", self._config)

    def finish(self) -> None:
        if self._summary:
            self._local.write_json("summary.json", self._summary)
        if self._enabled:
            self._wandb.finish()
