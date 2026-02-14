"""Experiment tracking abstraction.

Wraps wandb so that all other modules can log metrics without importing wandb directly.
When disabled (wandb config is None), all operations are no-ops.
"""

import logging

logger = logging.getLogger(__name__)


class MetricsTracker:
    def __init__(self, wandb_config: dict = None):
        """Initialize the tracker.

        Args:
            wandb_config: The 'wandb' block from the YAML config.
                          If None, tracking is disabled (no-op mode).
        """
        self._enabled = wandb_config is not None
        self._run = None

        if self._enabled:
            import wandb

            self._wandb = wandb
            wandb.init(config={}, **wandb_config)
            self._run = wandb.run
        else:
            self._wandb = None

    def log(self, metrics: dict):
        """Log a dictionary of metrics."""
        if self._enabled:
            self._wandb.log(metrics)

    def log_table(self, name: str, columns: list, data: list):
        """Log a wandb Table."""
        if self._enabled:
            table = self._wandb.Table(columns=columns, data=data)
            self._wandb.log({name: table})

    def log_plot_table(self, name: str, vega_spec_name: str, data_table, fields: dict):
        """Log a wandb plot table (e.g. tree visualization)."""
        if self._enabled:
            tree = self._wandb.plot_table(
                vega_spec_name=vega_spec_name,
                data_table=data_table,
                fields=fields,
            )
            self._wandb.log({name: tree})

    def create_table(self, columns: list, data: list):
        """Create a wandb Table object (for use with log_plot_table)."""
        if self._enabled:
            return self._wandb.Table(columns=columns, data=data)
        return None

    def set_summary(self, key: str, value):
        """Set a summary metric on the run."""
        if self._enabled and self._run is not None:
            self._run.summary[key] = value

    def set_config(self, config: dict):
        """Update the run config after initialization."""
        if self._enabled and self._run is not None:
            self._run.config.update(config)

    def finish(self):
        """Finish the tracking run."""
        if self._enabled:
            self._wandb.finish()
