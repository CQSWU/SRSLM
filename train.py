import sys

from copy import deepcopy
from pathlib import Path


import torch

import yaml

from sample_factory.algo.utils.context import global_env_registry

from sample_factory.cfg.arguments import default_cfg

from sample_factory.train import run_rl

from sample_factory.utils.utils import log





from learning.config import Environment, Experiment


import learning.encoder


from pomapf_env.env import make_pomapf

from pomapf_env.wrappers import MatrixObservationWrapper, TauObservationWrapper


_CUSTOM_COMPONENTS_REGISTERED = False

_LEGACY_CONFIG_KEYS = {
    'async_ppo': {
        'num_minibatches_to_accumulate',
        'traj_buffers_excess_ratio',
        'reset_timeout_seconds',
        'train_in_background_thread',
        'learner_main_loop_num_cores',
        'pbt_optimize_batch_size',
        'use_cpc',
        'cpc_forward_steps',
        'cpc_time_subsample',
        'cpc_forward_subsample',
        'sampler_only',
    },
    'experiment_settings': {
        'caar_freeze_backbone',
        'encoder_type',
        'use_spectral_norm',
        'warm_start_from',
    },
    'global_settings': {
        'experiments_root',
        'use_wandb',
    },
    'evaluation': {
        'render_action_repeat',
        'record_to',
        'continuous_actions_sample',
        'eval_config',
    },
}


def _patch_checkpoint_loading():

    from sample_factory.algo.learning.learner import Learner

    if getattr(Learner.load_checkpoint, '_caar_checkpoint_patch', False):

        return

    def patched(checkpoints, device):

        if not checkpoints:

            log.warning('No checkpoints found')

            return None

        latest_checkpoint = checkpoints[-1]

        device_type = getattr(device, 'type', str(device))

        load_device = torch.device('cpu') if device_type == 'mps' else device

        last_error = None
        for attempt in range(3):

            try:

                log.warning('Loading state from checkpoint %s...', latest_checkpoint)

                try:

                    return torch.load(

                        latest_checkpoint,

                        map_location=load_device,

                        weights_only=False,

                    )

                except TypeError:

                    return torch.load(latest_checkpoint, map_location=load_device)

            except Exception as error:

                log.exception(
                    'Could not load from checkpoint, attempt %d of 3',
                    attempt + 1,
                )
                last_error = error

        raise RuntimeError(
            f'Could not load checkpoint after 3 attempts: {latest_checkpoint}'
        ) from last_error

    patched._caar_checkpoint_patch = True

    Learner.load_checkpoint = staticmethod(patched)


_patch_checkpoint_loading()


def make_env(env_cfg: Environment | None = None):

    if env_cfg is None:
        env_cfg = Environment()
    return make_pomapf(grid_config=env_cfg.grid_config)



def create_pogema_env(full_env_name, cfg=None, env_config=None, render_mode=None):
    del env_config, render_mode

    _ensure_patched()

    environment_config: Environment = Environment(**cfg.full_config['environment'])

    env = make_env(environment_config)

    env = MatrixObservationWrapper(env)

    if full_env_name == 'POMAPF-ST-v0':
        env = TauObservationWrapper(
            env,
            rho=environment_config.tau_rho,
            tau_radius=environment_config.tau_radius,
        )

    return env



def _ensure_patched():

    import sample_factory.algo.utils.make_env as make_env_mod


    if getattr(
        make_env_mod.get_multiagent_info,
        '_pomapf_multiagent_patch',
        False,
    ):

        return


    def patched(env):

        current = env

        while True:

            attrs = getattr(current, '__dict__', {})

            if 'num_agents' in attrs and 'is_multiagent' in attrs:

                return attrs['is_multiagent'], attrs['num_agents']

            if hasattr(current, 'env') and current.env is not current:

                current = current.env

            else:

                break

        return getattr(env, 'is_multiagent', False), getattr(env, 'num_agents', 1)


    patched._pomapf_multiagent_patch = True
    make_env_mod.get_multiagent_info = patched

    make_env_mod.is_multiagent_env = lambda env: patched(env)[0]



def register_custom_components():

    global _CUSTOM_COMPONENTS_REGISTERED

    if _CUSTOM_COMPONENTS_REGISTERED:

        return


    global_env_registry()['POMAPF-v0'] = create_pogema_env

    global_env_registry()['POMAPF-ST-v0'] = create_pogema_env

    _ensure_patched()

    _patch_checkpoint_loading()

    _CUSTOM_COMPONENTS_REGISTERED = True



def _migrate_legacy_config(config):
    """Translate serialized pre-cleanup configs without weakening YAML validation."""
    if not isinstance(config, dict):
        raise ValueError('Training config must be a mapping')
    migrated = deepcopy(config)
    async_config = migrated.get('async_ppo')
    global_config = migrated.get('global_settings')
    evaluation_config = migrated.get('evaluation')
    is_serialized_legacy_config = (
        isinstance(async_config, dict)
        and 'num_minibatches_to_accumulate' in async_config
        and isinstance(global_config, dict)
        and 'experiments_root' in global_config
        and isinstance(evaluation_config, dict)
        and 'record_to' in evaluation_config
    )
    if not is_serialized_legacy_config:
        return migrated

    if 'ppo_epochs' in async_config:
        async_config.setdefault('num_epochs', async_config['ppo_epochs'])
        async_config.pop('ppo_epochs')

    for section, keys in _LEGACY_CONFIG_KEYS.items():
        section_config = migrated.get(section)
        if not isinstance(section_config, dict):
            continue
        for key in keys:
            section_config.pop(key, None)
    return migrated


def validate_config(config):

    exp = Experiment(**_migrate_legacy_config(config))
    train_dir = Path(exp.global_settings.train_dir).expanduser()
    if not train_dir.is_absolute():
        train_dir = Path(__file__).resolve().parent / train_dir
    exp.global_settings.train_dir = str(train_dir.resolve())

    flat_config = default_cfg(

        algo=exp.global_settings.algo,

        env=exp.environment.name,

        experiment=exp.name or '',

    )

    for settings in (exp.async_ppo, exp.experiment_settings, exp.global_settings, exp.evaluation):

        for key, value in settings.dict().items():

            setattr(flat_config, key, value)

    flat_config.num_batches_per_epoch = exp.async_ppo.num_batches_per_iteration

    flat_config.rnn_size = exp.experiment_settings.hidden_size

    flat_config.encoder_conv_architecture = exp.experiment_settings.encoder_subtype

    flat_config.encoder_conv_mlp_layers = [exp.experiment_settings.hidden_size]

    flat_config.full_config = exp.dict()

    return exp, flat_config



def main():

    import argparse


    parser = argparse.ArgumentParser(description='Process training config.')

    parser.add_argument(

        '--config_path',

        type=str,

        action='store',

        help='path to yaml file with single run configuration',

        required=True,

    )

    params = parser.parse_args()


    register_custom_components()

    with open(params.config_path, 'r', encoding='utf-8') as f:

        config = yaml.safe_load(f)


    exp, flat_config = validate_config(config)

    status = run_rl(flat_config)

    return status



if __name__ == '__main__':

    sys.exit(main())
