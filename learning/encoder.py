from sample_factory.algo.utils.context import global_model_factory
from sample_factory.model.actor_critic import default_make_actor_critic_func
from sample_factory.model.encoder import default_make_encoder_func


def _encoder_kind(cfg):
    kind = getattr(cfg, "encoder_custom", None)
    if kind not in {
        None,
        "pogema_residual",
        "epom_finetune",
        "epom_trace_context",
        "switcher",
        "switcher_all_state",
    }:
        raise ValueError(f"Unsupported or retired encoder_custom: {kind!r}")
    return kind


def make_encoder(cfg, obs_space):
    kind = _encoder_kind(cfg)

    if kind in (
        "switcher",
        "switcher_all_state",
    ):
        from learning.switcher_actor_critic import SwitcherEncoder

        return SwitcherEncoder(cfg, obs_space)

    if kind in (
        "pogema_residual",
        "epom_trace_context",
        "epom_finetune",
    ):
        from learning.epom_encoder import EPOMEncoder

        return EPOMEncoder(cfg, obs_space)

    return default_make_encoder_func(cfg, obs_space)


def make_actor_critic(cfg, obs_space, action_space):
    kind = _encoder_kind(cfg)

    if kind == "epom_finetune":
        from learning.epom_finetune_actor_critic import EPOMFineTuneActorCritic

        actor_critic = EPOMFineTuneActorCritic

    elif kind == "switcher_all_state":
        from learning.switcher_actor_critic import AllStateSwitcherActorCritic

        actor_critic = AllStateSwitcherActorCritic

    elif kind == "switcher":
        from learning.switcher_actor_critic import SwitcherActorCritic

        actor_critic = SwitcherActorCritic

    elif kind == "epom_trace_context":
        from learning.epom_trace_multiplier_actor_critic import (
            EPOMTraceMultiplierActorCritic,
        )

        actor_critic = EPOMTraceMultiplierActorCritic
    else:
        return default_make_actor_critic_func(cfg, obs_space, action_space)

    return actor_critic(global_model_factory(), obs_space, action_space, cfg)


global_model_factory().register_encoder_factory(make_encoder)
global_model_factory().register_actor_critic_factory(make_actor_critic)
