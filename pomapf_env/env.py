from pogema.envs import (

    CSRMetric,

    EpLengthMetric,

    ISRMetric,

    LifeLongAverageThroughputMetric,

    MultiTimeLimit,

    NonDisappearCSRMetric,

    NonDisappearEpLengthMetric,

    NonDisappearISRMetric,

    Pogema,

    PogemaCoopFinish,

    PogemaLifeLong,

    SumOfCostsAndMakespanMetric,

)

from pogema.integrations.sample_factory import AutoResetWrapper, IsMultiAgentWrapper, MetricsForwardingWrapper

from pogema.svg_animation.animation_wrapper import AnimationConfig, AnimationMonitor

from pogema.wrappers.persistence import PersistentWrapper

from copy import deepcopy


from pomapf_env.wrappers import EnvAttributesWrapper, RewardShaping, MultiMapWrapper




import gymnasium as _gymnasium


if not hasattr(_gymnasium.logger, 'info'):

    def _logger_info(msg, *args):

        print(msg % args if args else msg)

    _gymnasium.logger.info = _logger_info



def _patch_persistent_wrapper():

    """PersistentWrapper assumes old-gym attribute forwarding which gymnasium removed."""

    if hasattr(PersistentWrapper, '_patched_for_gymnasium'):

        return


    def get_num_agents(self):

        return self.env.get_num_agents()


    @property

    def grid(self):

        return self.env.grid


    @property

    def grid_config(self):

        return self.env.grid_config


    def set_elapsed_steps(self, steps):

        return self.env.set_elapsed_steps(steps)


    def get_obstacles(self, ignore_borders=False):

        return self.env.get_obstacles(ignore_borders=ignore_borders)


    PersistentWrapper.get_num_agents = get_num_agents

    PersistentWrapper.grid = grid

    PersistentWrapper.grid_config = grid_config

    PersistentWrapper.set_elapsed_steps = set_elapsed_steps

    PersistentWrapper.get_obstacles = get_obstacles

    PersistentWrapper._patched_for_gymnasium = True



def _patch_animation_monitor():

    if hasattr(AnimationMonitor, '_patched_for_gymnasium'):

        return


    @property

    def grid_config(self):

        return self.env.grid_config


    AnimationMonitor.grid_config = grid_config

    AnimationMonitor._patched_for_gymnasium = True



_patch_persistent_wrapper()

_patch_animation_monitor()



class _MetricCompatMixin:

    """Restore direct metadata access for Pogema metrics under Gymnasium wrappers."""


    @property

    def grid_config(self):

        return self.env.unwrapped.grid_config


    def get_num_agents(self):

        return self.grid_config.num_agents


    @property

    def was_on_goal(self):

        return self.env.unwrapped.was_on_goal



class _MultiTimeLimit(_MetricCompatMixin, MultiTimeLimit):

    pass



class _CSRMetric(_MetricCompatMixin, CSRMetric):

    pass



class _EpLengthMetric(_MetricCompatMixin, EpLengthMetric):

    pass



class _ISRMetric(_MetricCompatMixin, ISRMetric):

    pass



class _LifeLongAverageThroughputMetric(_MetricCompatMixin, LifeLongAverageThroughputMetric):

    pass



class _NonDisappearCSRMetric(_MetricCompatMixin, NonDisappearCSRMetric):

    pass



class _NonDisappearEpLengthMetric(_MetricCompatMixin, NonDisappearEpLengthMetric):

    pass



class _NonDisappearISRMetric(_MetricCompatMixin, NonDisappearISRMetric):

    pass



class _SumOfCostsAndMakespanMetric(_MetricCompatMixin, SumOfCostsAndMakespanMetric):

    pass



def _make_pogema_base(grid_config):

    if grid_config.on_target == 'restart':

        env = PogemaLifeLong(grid_config=grid_config)

    elif grid_config.on_target == 'nothing':

        env = PogemaCoopFinish(grid_config=grid_config)

    elif grid_config.on_target == 'finish':

        env = Pogema(grid_config=grid_config)

    else:

        raise KeyError(f'Unknown on_target option: {grid_config.on_target}')


    env = _MultiTimeLimit(env, grid_config.max_episode_steps)

    if grid_config.persistent:

        return PersistentWrapper(env)


    if grid_config.on_target == 'restart':

        return _LifeLongAverageThroughputMetric(env)

    if grid_config.on_target == 'nothing':

        env = _NonDisappearISRMetric(env)

        env = _NonDisappearCSRMetric(env)

        env = _NonDisappearEpLengthMetric(env)

        return _SumOfCostsAndMakespanMetric(env)

    if grid_config.on_target == 'finish':

        env = _ISRMetric(env)

        env = _CSRMetric(env)

        return _EpLengthMetric(env)

    raise KeyError(f'Unknown on_target option: {grid_config.on_target}')



def _make_sample_factory_pogema(grid_config):

    env = _make_pogema_base(grid_config)

    env = MetricsForwardingWrapper(env)

    env = IsMultiAgentWrapper(env)

    return env



def make_pomapf(grid_config, with_animations=False, auto_reset=True):

    grid_config = deepcopy(grid_config)

    grid_config.auto_reset = False

    env = _make_sample_factory_pogema(grid_config)

    if with_animations:

        env = AnimationMonitor(env, AnimationConfig(egocentric_idx=0))

    env = RewardShaping(env)


    env = MultiMapWrapper(env)

    if auto_reset:

        env = AutoResetWrapper(env)

    return EnvAttributesWrapper(env)
