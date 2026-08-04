"""Raw AO-RePlan proposals for learning-guided hybrid policies.

This module intentionally never constructs or calls a Probe.  It exposes the
dynamic planner's proposal together with a reverse mask and requires callers
to commit exactly the proposals that were actually sent to the environment.
The standalone :mod:`agents.ao_replan` policy keeps its historical behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from pogema import GridConfig

from planning.ao_replan_algo import AORePlanBase, INF


@dataclass(frozen=True)
class RawPlanBatch:
    """One pending batch of unmodified dynamic-planner proposals."""

    actions: tuple[int | None, ...]
    planned_mask: tuple[bool, ...]
    reverse_mask: tuple[bool, ...]


class RawAORePlanCandidates:
    """Generate raw Plan actions without allowing Probe substitution.

    A reverse is defined from *executed positions*: after an agent has moved
    from ``A`` to ``B``, a proposal that would move it from ``B`` back to
    ``A`` is a reverse.  Wait, ``None``, blocked moves, and a new target do not
    create a reverse event.
    """

    def __init__(
        self,
        *,
        use_best_move: bool = True,
        max_steps: int = INF,
        seed: int | None = None,
        base_factory: Callable[..., AORePlanBase] = AORePlanBase,
    ):
        self.use_best_move = bool(use_best_move)
        self.max_steps = int(max_steps)
        self.seed = seed
        self._base_factory = base_factory
        self._moves = tuple(
            tuple(int(value) for value in move)
            for move in GridConfig().MOVES
        )
        self.reset()

    def reset(self) -> None:
        self._base = self._base_factory(
            use_best_move=self.use_best_move,
            max_steps=self.max_steps,
            seed=self.seed,
        )
        self._position_history: list[list[tuple[int, int]]] | None = None
        self._last_target: list[tuple[int, int] | None] | None = None
        self._pending: RawPlanBatch | None = None

    @staticmethod
    def _point(observation, key: str) -> tuple[int, int]:
        value = observation[key]
        return int(value[0]), int(value[1])

    def _ensure_state(self, count: int) -> None:
        if self._position_history is None:
            self._position_history = [[] for _ in range(count)]
            self._last_target = [None] * count
            return
        if len(self._position_history) != count:
            raise ValueError(
                "Raw AO-RePlan candidate count changed without reset()."
            )

    def _is_reverse(self, index: int, position, action) -> bool:
        if action in (None, 0):
            return False
        history = self._position_history[index]
        if len(history) < 2:
            return False
        dx, dy = self._moves[int(action)]
        return (position[0] + dx, position[1] + dy) == history[-2]

    def propose(
        self,
        observations: Sequence,
        *,
        skip_agents: Sequence[bool] | None = None,
    ) -> RawPlanBatch:
        """Return raw actions and hold their planner feedback pending commit."""

        if self._pending is not None:
            raise RuntimeError(
                "The previous raw Plan batch must be committed before propose()."
            )
        count = len(observations)
        if skip_agents is None:
            skip = [False] * count
        else:
            if len(skip_agents) != count:
                raise ValueError(
                    "skip_agents and observations must have equal sizes."
                )
            skip = [bool(value) for value in skip_agents]

        # Passing an explicit skip mask selects AORePlanBase's synchronized
        # proposal/commit protocol even when no agent is skipped.
        actions = self._base.act(observations, skip_agents=skip)
        if len(actions) != count:
            raise RuntimeError("AO-RePlan returned the wrong action count.")

        self._ensure_state(count)
        reverse_mask = [False] * count
        for index, observation in enumerate(observations):
            target = self._point(observation, "target_xy")
            if self._last_target[index] != target:
                self._position_history[index].clear()
                self._last_target[index] = target

            position = self._point(observation, "xy")
            history = self._position_history[index]
            if not history or history[-1] != position:
                history.append(position)
                if len(history) > 2:
                    del history[:-2]

            if not skip[index]:
                reverse_mask[index] = self._is_reverse(
                    index,
                    position,
                    actions[index],
                )

        batch = RawPlanBatch(
            actions=tuple(
                None if action is None else int(action)
                for action in actions
            ),
            planned_mask=tuple(action is not None for action in actions),
            reverse_mask=tuple(reverse_mask),
        )
        self._pending = batch
        return batch

    def commit(self, executed_mask: Sequence[bool]) -> None:
        """Commit executed Plan proposals and cancel every other proposal."""

        if self._pending is None:
            raise RuntimeError("propose() must be called before commit().")
        if len(executed_mask) != len(self._pending.actions):
            raise ValueError(
                "executed_mask and the pending Plan batch must have equal sizes."
            )
        mask = [bool(value) for value in executed_mask]
        if any(
            executed and not planned
            for executed, planned in zip(mask, self._pending.planned_mask)
        ):
            raise ValueError("A missing raw Plan proposal cannot be committed.")
        if any(
            executed and reverse
            for executed, reverse in zip(mask, self._pending.reverse_mask)
        ):
            raise ValueError(
                "A raw reverse Plan proposal must fall back to CAAR."
            )

        self._base.commit_proposals(mask)
        self._pending = None

    @property
    def pending(self) -> RawPlanBatch | None:
        return self._pending
