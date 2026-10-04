from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from planning.ao_replan_algo import AORePlanBase, AORePlanWrapper, INF


@dataclass(frozen=True)
class AORePlanStep:
    actions: tuple[int | None, ...]
    planned_mask: tuple[bool, ...]
    reverse_mask: tuple[bool, ...]
    static_astar_invoked_mask: tuple[bool, ...]


class AORePlanBranch:
    def __init__(
        self,
        *,
        max_steps: int = INF,
        seed: int | None = None,
        base_factory: Callable[..., AORePlanBase] = AORePlanBase,
        wrapper_factory: Callable[..., AORePlanWrapper] = AORePlanWrapper,
    ):
        self.max_steps = int(max_steps)
        self.seed = seed
        self._base_factory = base_factory
        self._wrapper_factory = wrapper_factory
        self.reset()

    def reset(self) -> None:
        self._base = self._base_factory(
            max_steps=self.max_steps,
            seed=self.seed,
        )
        self._wrapper = self._wrapper_factory(
            self._base,
            max_steps=self.max_steps,
        )
        self._pending: AORePlanStep | None = None

    @staticmethod
    def _optional_int_tuple(values):
        return tuple(None if value is None else int(value) for value in values)

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
        if skip_agents is None:
            skip = [False] * count
        else:
            if len(skip_agents) != count:
                raise ValueError("skip_agents and observations must have equal sizes.")
            skip = [bool(value) for value in skip_agents]

        actions = list(self._wrapper.act(observations, skip_agents=skip))
        fields = (
            self._wrapper.last_reverse_mask,
            self._wrapper.last_static_astar_invoked_mask,
        )
        if len(actions) != count or any(
            values is None or len(values) != count for values in fields
        ):
            raise RuntimeError("AORePlan returned an inconsistent candidate batch.")

        batch = AORePlanStep(
            actions=self._optional_int_tuple(actions),
            planned_mask=tuple(action is not None for action in actions),
            reverse_mask=tuple(bool(value) for value in fields[0]),
            static_astar_invoked_mask=tuple(bool(value) for value in fields[1]),
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
        physical = [bool(value) for value in executed_mask]
        if any(
            matched and not planned
            for matched, planned in zip(physical, self._pending.planned_mask)
        ):
            raise ValueError("A missing AORePlan proposal cannot be committed.")

        overridden = tuple(
            bool(value) for value in self._wrapper.last_dynamic_override_mask
        )
        base_mask = [
            matched and not replaced for matched, replaced in zip(physical, overridden)
        ]
        self._base.commit_proposals(base_mask)
        self._pending = None


__all__ = ["AORePlanBranch", "AORePlanStep"]
