from sample_factory.algo.utils.context import global_model_factory
from sample_factory.model.actor_critic import default_make_actor_critic_func
from sample_factory.model.encoder import default_make_encoder_func



def make_encoder(cfg, obs_space):

    if getattr(cfg, 'encoder_custom', None) in (
        'switcher',
        'switcher_all_state',
    ):

        from learning.switcher_actor_critic import SwitcherEncoder

        return SwitcherEncoder(cfg, obs_space)

    if getattr(cfg, 'encoder_custom', None) == 'caar_ra_gate':

        from learning.caar_ra_actor_critic import CAARRAGateEncoder

        return CAARRAGateEncoder(cfg, obs_space)

    if getattr(cfg, 'encoder_custom', None) in (
        'pogema_residual',
        'epom_trace',
        'epom_trace_residual',
        'epom_trace_context',
        'epom_finetune',
    ):

        from learning.epom_encoder import EPOMEncoder

        return EPOMEncoder(cfg, obs_space)

    if getattr(cfg, 'encoder_custom', None) == 'caar':

        from learning.caar_encoder import CAAREncoder

        return CAAREncoder(cfg, obs_space)

    if getattr(cfg, 'encoder_custom', None) == 'trace_residual':

        from learning.caar_encoder import CAAREncoder

        return CAAREncoder(cfg, obs_space)

    if getattr(cfg, 'encoder_custom', None) == 'mast':

        from learning.mast_encoder import MASTEncoder

        return MASTEncoder(cfg, obs_space)

    return default_make_encoder_func(cfg, obs_space)


def make_actor_critic(cfg, obs_space, action_space):

    if getattr(cfg, 'encoder_custom', None) == 'trace_residual':

        from learning.trace_residual_actor_critic import TraceResidualActorCritic

        return TraceResidualActorCritic(

            global_model_factory(),

            obs_space,

            action_space,

            cfg,

        )

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

    if getattr(cfg, 'encoder_custom', None) == 'epom_trace_residual':

        from learning.epom_trace_residual_actor_critic import (
            EPOMTraceResidualActorCritic,
        )

        return EPOMTraceResidualActorCritic(
            global_model_factory(), obs_space, action_space, cfg,
        )

    if getattr(cfg, 'encoder_custom', None) == 'epom_trace_context':

        if getattr(cfg, 'trace_context_architecture', 'context') in (
            'multiplier',
            'coefficient',
            'scalar_gate',
            'factorized_gate',
            'entropy_scalar',
            'entropy_direction',
            'tiny_residual16',
            'linear_spatial_residual',
            'linear_gain',
            'conv_residual64',
            'conv_residual_linear',
            'conv_residual32',
            'conv_residual64_p_only',
            'conv_residual64_hlinear_critic',
            'conv_residual64_hmlp_critic',
            'conv_residual64_linear_value_critic',
            'paper_entropy_multiplier',
            'paper_entropy_fusion',
        ):

            from learning.epom_trace_multiplier_actor_critic import (
                EPOMTraceMultiplierActorCritic,
            )

            return EPOMTraceMultiplierActorCritic(
                global_model_factory(), obs_space, action_space, cfg,
            )

        from learning.epom_trace_context_actor_critic import (
            EPOMTraceContextActorCritic,
        )

        return EPOMTraceContextActorCritic(
            global_model_factory(), obs_space, action_space, cfg,
        )

    if getattr(cfg, 'encoder_custom', None) == 'epom_trace':

        from learning.epom_trace_actor_critic import EPOMTraceActorCritic

        return EPOMTraceActorCritic(

            global_model_factory(),

            obs_space,

            action_space,

            cfg,

        )

    if getattr(cfg, 'encoder_custom', None) == 'caar_ra_gate':

        from learning.caar_ra_actor_critic import CAARRAActorCritic

        return CAARRAActorCritic(

            global_model_factory(),

            obs_space,

            action_space,

            cfg,

        )

    if getattr(cfg, 'encoder_custom', None) == 'caar' and 'tau' in obs_space.spaces:

        from learning.caar_actor_critic import CAARActorCritic

        return CAARActorCritic(

            global_model_factory(),

            obs_space,

            action_space,

            cfg,

        )

    return default_make_actor_critic_func(cfg, obs_space, action_space)



global_model_factory().register_encoder_factory(make_encoder)
global_model_factory().register_actor_critic_factory(make_actor_critic)
