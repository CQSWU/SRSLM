"""Explicit standalone soft ablation; never used by SRSLM's planner branch."""
from agents.ao_replan import AORePlan
from planning.ao_replan_algo import AORePlanWrapper


class _SoftNoCheckWrapper(AORePlanWrapper):
    def _static_astar_action(self, index, observation):
        raw_action = self.static_astar.get_action(observation)
        self.last_static_astar_invoked_mask[index] = True
        # The soft simulator resolves conflicts; no local occupied-cell wait.
        return 0 if raw_action is None else int(raw_action)


class AORePlanSoftNoCheck(AORePlan):
    WRAPPER_CLASS = _SoftNoCheckWrapper

    def set_grid_config(self, grid_config):
        if getattr(grid_config, "collision_system", None) != "soft":
            raise ValueError("AORePlan-SoftNoCheck is only the standalone soft ablation.")
        super().set_grid_config(grid_config)
