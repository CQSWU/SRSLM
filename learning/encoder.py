from sample_factory.algo.utils.context import global_model_factory
from sample_factory.model.actor_critic import default_make_actor_critic_func
from sample_factory.model.encoder import default_make_encoder_func


def _require_retained_encoder(cfg, obs_space=None):
    kind = getattr(cfg, 'encoder_custom', None)
    if kind not in {None, 'pogema_residual', 'epom_finetune', 'caar',
                    'epom_trace_context', 'switcher', 'switcher_all_state'}:
        raise ValueError(f'Unsupported or retired encoder_custom: {kind!r}')
    if kind == 'caar' and obs_space is not None and 'tau' in obs_space.spaces:
        raise ValueError(
            'The caar+tau actor is retired. The retained caar encoder is '
            'the no-tau NoReweight backbone; use epom_trace_context for ARPE.'
        )


def make_encoder(cfg, obs_space):
    _require_retained_encoder(cfg, obs_space)

    if getattr(cfg, 'encoder_custom', None) in (
        'switcher',
        'switcher_all_state',
    ):

        from learning.switcher_actor_critic import SwitcherEncoder

        return SwitcherEncoder(cfg, obs_space)

    if getattr(cfg, 'encoder_custom', None) in (
        'pogema_residual',
        'epom_trace_context',
        'epom_finetune',
    ):

        from learning.epom_encoder import EPOMEncoder

        return EPOMEncoder(cfg, obs_space)

    if getattr(cfg, 'encoder_custom', None) == 'caar':

        from learning.no_reweight_encoder import NoReweightEncoder

        return NoReweightEncoder(cfg, obs_space)

    return default_make_encoder_func(cfg, obs_space)


def make_actor_critic(cfg, obs_space, action_space):
    _require_retained_encoder(cfg, obs_space)

    if getattr(cfg, 'encoder_custom', None) == 'epom_finetune':

        from learning.epom_finetune_actor_critic import EPOMFineTuneActorCritic

        return EPOMFineTuneActorCritic(

            global_model_factory(),

            obs_space,

            action_space,

            cfg,

        )

    if getattr(cfg, 'encoder_custom', None) == 'switcher_all_state':

        from learning.switcher_actor_critic import AllStateSwitcherActorCritic

        return AllStateSwitcherActorCritic(

            global_model_factory(),

            obs_space,

            action_space,

            cfg,

        )

    if getattr(cfg, 'encoder_custom', None) == 'switcher':

        from learning.switcher_actor_critic import SwitcherActorCritic

        return SwitcherActorCritic(

            global_model_factory(),

            obs_space,

            action_space,

            cfg,

        )

    if getattr(cfg, 'encoder_custom', None) == 'epom_trace_context':
        from learning.epom_trace_multiplier_actor_critic import (
            EPOMTraceMultiplierActorCritic,
        )

        return EPOMTraceMultiplierActorCritic(
            global_model_factory(), obs_space, action_space, cfg,
        )

    return default_make_actor_critic_func(cfg, obs_space, action_space)



global_model_factory().register_encoder_factory(make_encoder)
global_model_factory().register_actor_critic_factory(make_actor_critic)
