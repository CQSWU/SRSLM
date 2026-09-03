from typing import Literal


from pydantic import Extra, Field

from pogema import GridConfig


from agents.utils_agents import AlgoBase
from agents.reverse_metrics import ExecutedPositionReverseCounter

from planning.replan_algo import RePlanCore, FixLoopsWrapper, NoPathSoRandomOrStayWrapper, FixNonesWrapper



class RePlanConfig(AlgoBase, extra=Extra.forbid):

    name: Literal['RePlan'] = 'RePlan'

    fix_loops: bool = True

    no_path_random: bool = True

    fix_nones: bool = True

    add_none_if_loop: bool = True

    use_best_move: bool = True

    stay_if_loop_prob: float = Field(0.5, ge=0.0, le=1.0)

    max_planning_steps: int = Field(10000, gt=0)

    device: str = 'cpu'



class RePlan:

    def __init__(self, cfg: RePlanConfig):

        self.cfg = cfg

        self.agent = None

        self.fix_loops = cfg.fix_loops

        self.fix_nones = cfg.fix_nones

        self.stay_if_loop_prob = cfg.stay_if_loop_prob

        self.no_path_random = cfg.no_path_random

        self.use_best_move = cfg.use_best_move

        self.add_none_if_loop = cfg.add_none_if_loop

        self._reverse_counter = ExecutedPositionReverseCounter(GridConfig().MOVES)


    def act(self, observations, rewards=None, dones=None, info=None, skip_agents=None):

        actions = self.agent.act(observations, skip_agents)

        self._reverse_counter.record(actions, observations)

        return actions


    @property

    def reverse_action_rate(self):

        return self._reverse_counter.rate

    @property
    def reverse_action_count(self):

        return self._reverse_counter.reverse_count

    @property
    def reverse_action_denominator(self):

        return self._reverse_counter.movement_count

    @property
    def reverse_metric_version(self):

        return self._reverse_counter.METRIC_VERSION


    def after_step(self, dones):

        if all(dones):

            self.agent = None


    def after_reset(self, ):

        self.agent = RePlanCore(
            use_best_move=self.use_best_move,
            max_steps=self.cfg.max_planning_steps,
            seed=self.cfg.seed,
        )


        if self.fix_loops:

            self.agent = FixLoopsWrapper(self.agent, stay_if_loop_prob=self.stay_if_loop_prob,

                                         add_none_if_loop=self.add_none_if_loop)

        if self.no_path_random:

            self.agent = NoPathSoRandomOrStayWrapper(self.agent)

        elif self.fix_nones:

            self.agent = FixNonesWrapper(self.agent)

        self._reverse_counter.reset()
