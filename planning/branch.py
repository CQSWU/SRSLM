from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from planning.aoreplan import AORePlanBase, AORePlanWrapper, INF


@dataclass(frozen=True)
class AORePlanStep:
    actions: tuple[int | None, ...]
    planned_mask: tuple[bool, ...]


class AORePlanBranch:
    def __init__(
        self,
        *,
        max_steps: int = INF,
        seed: int | None = None,
    ):
        self.max_steps = int(max_steps)
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        base = AORePlanBase(
            max_steps=self.max_steps,
            seed=self.seed,
        )
        self._wrapper = AORePlanWrapper(
            base,
            max_steps=self.max_steps,
        )
        self._pending: AORePlanStep | None = None

    def propose(
        self,
        observations: Sequence,
        *,
        skip_agents: Sequence[bool] | None = None,
    ) -> AORePlanStep:
        if self._pending is not None:
            raise RuntimeError(
                "The previous AORePlan batch must be committed before propose()."
            )
        count = len(observations)
        actions = self._wrapper.act(observations, skip_agents=skip_agents)
        if len(actions) != count:
            raise RuntimeError("AORePlan returned an inconsistent candidate batch.")

        batch = AORePlanStep(
            actions=tuple(actions),
            planned_mask=tuple(action is not None for action in actions),
        )
        self._pending = batch
        return batch

    def commit(
        self,
        executed_mask: Sequence[bool],
    ) -> None:

        if self._pending is None:
            raise RuntimeError("propose() must be called before commit().")
        count = len(self._pending.actions)
        if len(executed_mask) != count:
            raise ValueError(
                "executed_mask and the pending AORePlan batch must have equal sizes."
            )
        if any(
            matched and not planned
            for matched, planned in zip(executed_mask, self._pending.planned_mask)
        ):
            raise ValueError("A missing AORePlan proposal cannot be committed.")

        self._wrapper.commit_proposals(executed_mask)
        self._pending = None


__all__ = ["AORePlanBranch", "AORePlanStep"]
