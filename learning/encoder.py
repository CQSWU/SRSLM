from sample_factory.algo.utils.context import global_model_factory
from sample_factory.model.actor_critic import default_make_actor_critic_func
from sample_factory.model.encoder import default_make_encoder_func



def make_encoder(cfg, obs_space):

    if getattr(cfg, 'encoder_custom', None) == 'caar_ra_gate':

        from learning.caar_ra_actor_critic import CAARRAGateEncoder

        return CAARRAGateEncoder(cfg, obs_space)

    if getattr(cfg, 'encoder_custom', None) == 'pogema_residual':

        from learning.epom_encoder import EPOMEncoder

        return EPOMEncoder(cfg, obs_space)

    if getattr(cfg, 'encoder_custom', None) == 'caar':

        from learning.caar_encoder import CAAREncoder

        return CAAREncoder(cfg, obs_space)

    return default_make_encoder_func(cfg, obs_space)


def make_actor_critic(cfg, obs_space, action_space):

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
