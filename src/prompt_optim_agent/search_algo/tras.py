"""TRAS — Textual Regularization with Aggregated Signals.

Paper: Davari et al., *Stabilizing Black-Box Prompt Optimization with Textual
Regularization and Signal Aggregation*. Canadian AI 2026.
Repo:  https://github.com/rezazzr/TRAS

TRAS shares the MCTS search backbone with PromptAgent (our :class:`MCTS`
class) but augments the inner-loop *update signal* used to revise prompts.
Specifically, it adds two components to the standard textual-gradient
update:

  1. **Textual regularization** — a success-conditioned signal derived from
     correct predictions that prescribes preservation of working prompt
     components (Section 3.3). Activated after a warmup of
     ``positive_reinforcement_depth`` iterations.

  2. **Monte Carlo Signal Aggregation (MCSA)** — sample :math:`K` independent
     textual gradients and regularizers, then aggregate each into a single
     actionable directive (Section 3.4). Controlled by the
     ``gradient_sampling`` field (K).

Choose the backbone via YAML:

    search_algo: mcts     # PromptAgent baseline (textual gradient only)
    search_algo: tras     # TRAS: gradient + regularizer + MCSA

TRAS-specific settings live under ``world_model_setting`` in the YAML config:

    world_model_setting:
      positive_reinforcement_depth: 4    # tau_warmup in the paper
      gradient_sampling: 6               # K in MCSA

Paper terminology → code identifier:

  ==============================  ======================================================================
  Paper                           Code
  ==============================  ======================================================================
  textual gradient (``g_t``)      ``gradient_descent.cal_gradient(..., gradient_type="negative")``
  textual regularization (``r_t``) ``gradient_descent.cal_gradient(..., gradient_type="positive")``;
                                   activated via ``positive_reinforcement_depth``
  MCSA samples (``K``)            ``gradient_descent.gradient_sampling``
  ==============================  ======================================================================
"""

from __future__ import annotations

import warnings

from .mcts import MCTS


class TRAS(MCTS):
    """TRAS search algorithm.

    Thin subclass of :class:`MCTS` that labels the run as TRAS and validates
    that the configuration actually enables the two TRAS-specific signals
    (textual regularization and MCSA). If either is disabled the run is
    degenerate (equivalent to the PromptAgent baseline), and the user is
    warned so they know to use ``search_algo: mcts`` instead.
    """

    def __init__(self, task, world_model, **kwargs):
        super().__init__(task=task, world_model=world_model, **kwargs)
        self._check_tras_config(world_model)

    @staticmethod
    def _check_tras_config(world_model) -> None:
        gd = getattr(world_model, "gradient_descent", None)
        if gd is None:
            return

        depth = getattr(gd, "positive_reinforcement_depth", 0)
        samples = getattr(gd, "gradient_sampling", 1)

        if depth <= 0:
            warnings.warn(
                "TRAS expects textual regularization to be active "
                "(world_model_setting.positive_reinforcement_depth > 0); "
                "got 0. This run reduces to the PromptAgent baseline — "
                "use `search_algo: mcts` if that is intentional.",
                stacklevel=3,
            )
        if samples <= 1:
            warnings.warn(
                f"TRAS expects MCSA with K >= 2 samples "
                f"(world_model_setting.gradient_sampling > 1); got {samples}. "
                f"No variance reduction will occur.",
                stacklevel=3,
            )
