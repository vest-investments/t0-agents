"""Episode-level usage accumulator across many :class:`Run`s.

A *conversation* in t0-agents is not a single :class:`~t0_agents.core.run.Run`:
it is the sequence of ``agent.invoke(...)`` calls that share one TensorZero
``episode_id``.  Each invoke produces its own Run with its own
:class:`~t0_agents.core.types.RunUsage`; nothing aggregates across them.

:class:`ConversationUsage` is that aggregate.  It tracks two distinct numbers
the outer layer (e.g. a PaaS deciding when to compact or rotate a chat) needs at
the same time:

- **Billing** — the cumulative :class:`RunUsage` summed across every recorded
  Run.  This is what you bill / budget against.
- **Footprint** — the *current* full-conversation prompt size, as a **level**
  (the latest recorded Run's
  :meth:`~t0_agents.core.run.Run.context_footprint`), **not** a sum.  This is
  what you compare against the model's context window to decide whether to
  compact.  Summing prompt sizes would double-count the prefix re-sent each
  turn, so footprint is deliberately a replace, not an add.

The model's context window is resolved by TensorZero and is not knowable
in-process, so :meth:`remaining` and :meth:`fraction_used` take the window as a
caller argument (matching :meth:`Run.context_remaining`).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from t0_agents.core.types import RunUsage, Usage

if TYPE_CHECKING:
    from t0_agents.core.run import Run

logger = logging.getLogger(__name__)

__all__ = ["ConversationUsage"]


class ConversationUsage:
    """Accumulate usage across the Runs of one ``episode_id`` conversation.

    Construct one per conversation (keyed by ``episode_id``) and feed it each
    Run as it completes::

        conv = ConversationUsage(episode_id)
        run1 = await agent.invoke("hello", episode_id=episode_id).result()
        conv.record(run1)
        run2 = await agent.invoke("more", episode_id=episode_id).result()
        conv.record(run2)

        conv.total.input_tokens      # billing: run1 + run2 input tokens
        conv.context_footprint()     # level: run2's current prompt size
        conv.remaining(128_000)      # window - footprint.input_tokens
        conv.fraction_used(128_000)  # footprint.input_tokens / window

    The accumulator is intentionally light: it stores the running billing
    :class:`RunUsage` and the latest footprint level, plus a count of recorded
    Runs — not a per-Run history (callers that need history can keep their own
    list of Runs).
    """

    def __init__(self, episode_id: str) -> None:
        self._episode_id = episode_id
        self._total = RunUsage()
        self._footprint: Usage | None = None
        self._runs_recorded = 0

    @property
    def episode_id(self) -> str:
        """The conversation's TensorZero episode id (its identity / key)."""
        return self._episode_id

    @property
    def total(self) -> RunUsage:
        """Cumulative billing :class:`RunUsage` summed across recorded Runs.

        Sums ``input_tokens`` / ``output_tokens`` / ``cost`` and both
        ``provider_cache_*`` fields; ``cost_is_priced`` is the OR across Runs
        (priced if any recorded Run reported a price).  This is the billing /
        budget number — contrast with :meth:`context_footprint`, which is a
        level, not a sum.
        """
        return self._total

    @property
    def runs_recorded(self) -> int:
        """How many Runs have been folded in via :meth:`record`."""
        return self._runs_recorded

    def record(self, run: Run[Any]) -> None:
        """Fold one completed ``run`` into the conversation totals.

        - **Billing**: adds ``run.usage()`` into :attr:`total` (a no-op when
          the Run produced no inference and ``usage()`` is ``None``).
        - **Footprint**: replaces the level with ``run.context_footprint()``
          **only when that is not None** — a Run with no non-zero inference
          (a failure before the first turn, or a whole-response cache hit)
          leaves the prior footprint intact rather than blanking the live
          signal.  This mirrors how
          :meth:`~t0_agents.core.run.Run.context_footprint` already carries
          forward over cache-served zero turns within a single Run.

        Recording a Run whose ``episode_id`` differs from this conversation's
        is allowed but logged at debug level — billing is best-effort and the
        caller owns grouping.
        """
        run_episode = run.episode_id
        if run_episode is not None and run_episode != self._episode_id:
            logger.debug(
                "ConversationUsage(%s).record() got a Run from episode %s; "
                "folding it in anyway",
                self._episode_id,
                run_episode,
            )

        usage = run.usage()
        if usage is not None:
            self._total = RunUsage(
                input_tokens=self._total.input_tokens + usage.input_tokens,
                output_tokens=self._total.output_tokens + usage.output_tokens,
                cost=self._total.cost + usage.cost,
                provider_cache_read_input_tokens=(
                    self._total.provider_cache_read_input_tokens
                    + usage.provider_cache_read_input_tokens
                ),
                provider_cache_write_input_tokens=(
                    self._total.provider_cache_write_input_tokens
                    + usage.provider_cache_write_input_tokens
                ),
                cost_is_priced=(
                    self._total.cost_is_priced or usage.cost_is_priced
                ),
            )

        footprint = run.context_footprint()
        if footprint is not None:
            self._footprint = footprint

        self._runs_recorded += 1

    def context_footprint(self) -> Usage | None:
        """The current full-conversation prompt size, as a level.

        The footprint of the most recently recorded Run that had one — i.e.
        approximately the whole conversation re-sent on its latest turn, the
        number to compare against the model's context window.  ``None`` until a
        Run with a non-zero inference has been recorded.
        """
        return self._footprint

    def remaining(self, window: int) -> int | None:
        """Tokens left in a caller-supplied context ``window``.

        Computed as ``window - context_footprint().input_tokens``.  The window
        is a caller argument because the model / variant — and thus its context
        size — is resolved by TensorZero and not knowable in-process.  **Not
        clamped**: may be negative when the current prompt already exceeds the
        window, so the overflow magnitude reaches the caller (parity with
        :meth:`~t0_agents.core.run.Run.context_remaining`).  ``None`` until
        there is a footprint.
        """
        if self._footprint is None:
            return None
        return window - self._footprint.input_tokens

    def fraction_used(self, window: int) -> float | None:
        """Fraction of a caller-supplied context ``window`` the prompt occupies.

        ``context_footprint().input_tokens / window`` — e.g. ``0.82`` means the
        current prompt fills 82% of the window, a natural threshold for a
        "compact soon" decision.  May exceed ``1.0`` when over-window (not
        clamped, like :meth:`remaining`).  ``None`` until there is a footprint.

        Raises :class:`ValueError` when ``window <= 0`` (a fraction of a
        non-positive window is undefined) — unlike :meth:`remaining`, which is
        a subtraction and accepts any window.
        """
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        if self._footprint is None:
            return None
        return self._footprint.input_tokens / window
