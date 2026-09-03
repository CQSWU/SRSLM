import json
import os
import sys

from copy import deepcopy
from pathlib import Path


_LOCAL_PPU_SDK = Path("/usr/local/PPU_SDK")
if "PPU_SDK" not in os.environ and _LOCAL_PPU_SDK.is_dir():
    os.environ["PPU_SDK"] = str(_LOCAL_PPU_SDK)


import torch

import yaml

from sample_factory.algo.utils.context import global_env_registry

from sample_factory.cfg.arguments import default_cfg

from sample_factory.train import run_rl

from sample_factory.utils.utils import log





from learning.config import Environment, Experiment

from learning.switcher_learner_patch import patch_switcher_learner_losses


import learning.encoder


from pomapf_env.env import make_pomapf

from pomapf_env.wrappers import (
    GridMemoryObservationWrapper,
    MatrixObservationWrapper,
    TauObservationWrapper,
    TraceContextTeamRewardWrapper,
)


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

patch_switcher_learner_losses()


def make_env(env_cfg: Environment | None = None):

    if env_cfg is None:
        env_cfg = Environment()
    return make_pomapf(grid_config=env_cfg.grid_config)



def create_pogema_env(full_env_name, cfg=None, env_config=None, render_mode=None):
    del render_mode

    _ensure_patched()

    environment_config: Environment = Environment(**cfg.full_config['environment'])

    training_populations = (
        environment_config.training_num_agents_by_worker
    )
    if training_populations is not None:
        if isinstance(env_config, dict):
            worker_index = env_config.get('worker_index', 0)
        else:
            worker_index = getattr(env_config, 'worker_index', 0)
        if worker_index is None:
            worker_index = 0
        try:
            worker_index = int(worker_index)
        except (TypeError, ValueError) as error:
            raise ValueError(
                'Sample Factory env_config.worker_index must be an integer.'
            ) from error
        if worker_index < 0:
            raise ValueError(
                'Sample Factory env_config.worker_index must be non-negative.'
            )

        worker_grid_config = deepcopy(environment_config.grid_config)
        worker_grid_config.num_agents = training_populations[
            worker_index % len(training_populations)
        ]
        environment_config = environment_config.copy(
            update={'grid_config': worker_grid_config}
        )

    if full_env_name in (
        'POMAPF-SRSLM-v0',
        'POMAPF-SRSLM-NoWaitDetect-v0',
    ):
        from pomapf_env.switcher_env import (
            AllStateSwitcherEnv,
            SwitcherEnv,
        )

        environment_class = (
            AllStateSwitcherEnv
            if full_env_name == 'POMAPF-SRSLM-NoWaitDetect-v0'
            else SwitcherEnv
        )

        return environment_class(
            grid_config=environment_config.grid_config,
            caar_weights_path=(
                environment_config.switcher_caar_weights_path
            ),
            caar_checkpoint_kind=(
                environment_config.switcher_caar_checkpoint_kind
            ),
            caar_device=environment_config.switcher_caar_device,
            max_planning_steps=(
                environment_config.switcher_max_planning_steps
            ),
            team_reward_coefficient=(
                environment_config.switcher_team_reward_coefficient
            ),
            feature_schema=environment_config.switcher_feature_schema,
        )

    env = make_env(environment_config)

    if full_env_name in ('POMAPF-EPOM-v0', 'POMAPF-EPOM-ST-v0'):
        environment = cfg.full_config['environment']
        env = GridMemoryObservationWrapper(
            env,
            memory_radius=environment['grid_memory_obs_radius'],
        )

    env = MatrixObservationWrapper(env)

    if full_env_name in ('POMAPF-ST-v0', 'POMAPF-EPOM-ST-v0'):
        is_trace_context = (
            getattr(cfg, 'encoder_custom', None) == 'epom_trace_context'
        )
        env = TauObservationWrapper(
            env,
            rho=environment_config.tau_rho,
            tau_radius=environment_config.tau_radius,
            trace_variant=getattr(environment_config, 'trace_variant', 'real'),
            raw_tau=bool(getattr(environment_config, 'tau_raw', False)),
            variant_seed=getattr(cfg, 'seed', None),
            include_free_mask=is_trace_context,
        )
        if is_trace_context:
            env = TraceContextTeamRewardWrapper(
                env,
                coefficient=(
                    environment_config.trace_context_team_reward_coefficient
                ),
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

    # Sample Factory uses the multiprocessing "spawn" context.  Installing
    # this at module import covers spawned learner processes, while repeating
    # the idempotent call here also covers direct programmatic registration.
    patch_switcher_learner_losses()

    if _CUSTOM_COMPONENTS_REGISTERED:

        return


    global_env_registry()['POMAPF-v0'] = create_pogema_env

    global_env_registry()['POMAPF-ST-v0'] = create_pogema_env

    global_env_registry()['POMAPF-EPOM-v0'] = create_pogema_env

    global_env_registry()['POMAPF-EPOM-ST-v0'] = create_pogema_env

    global_env_registry()['POMAPF-SRSLM-v0'] = create_pogema_env

    global_env_registry()[
        'POMAPF-SRSLM-NoWaitDetect-v0'
    ] = create_pogema_env

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


def _sync_resume_cli_overrides(flat_config, override_keys):
    """Keep explicit runtime overrides when Sample Factory resumes a run."""

    override_keys = set(override_keys)
    if not override_keys:
        return

    cli_args = dict(getattr(flat_config, 'cli_args', {}) or {})
    for key in override_keys:
        cli_args[key] = getattr(flat_config, key)
    flat_config.cli_args = cli_args

    resume_config = (
        Path(flat_config.train_dir)
        / flat_config.experiment
        / 'config.json'
    )
    if not resume_config.is_file():
        return

    saved = json.loads(resume_config.read_text(encoding='utf-8'))
    changed = False

    for key in override_keys:
        value = getattr(flat_config, key)
        if saved.get(key) != value:
            saved[key] = value
            changed = True

    full_config = saved.get('full_config')
    if isinstance(full_config, dict):
        nested_updates = []
        if 'experiment' in override_keys:
            nested_updates.extend(
                [
                    (('name',), flat_config.experiment),
                    (
                        ('global_settings', 'experiment'),
                        flat_config.experiment,
                    ),
                ]
            )
        if 'seed' in override_keys:
            nested_updates.append(
                (('global_settings', 'seed'), flat_config.seed)
            )
        if 'train_dir' in override_keys:
            nested_updates.append(
                (('global_settings', 'train_dir'), flat_config.train_dir)
            )
        if 'train_for_env_steps' in override_keys:
            nested_updates.append(
                (
                    ('experiment_settings', 'train_for_env_steps'),
                    flat_config.train_for_env_steps,
                )
            )
        if 'training_population' in override_keys:
            nested_updates.append(
                (
                    ('environment', 'training_num_agents_by_worker'),
                    list(
                        flat_config.full_config['environment'][
                            'training_num_agents_by_worker'
                        ]
                    ),
                )
            )

        for path, value in nested_updates:
            target = full_config
            for key in path[:-1]:
                child = target.get(key)
                if not isinstance(child, dict):
                    child = {}
                    target[key] = child
                target = child
            if target.get(path[-1]) != value:
                target[path[-1]] = value
                changed = True

    if changed:
        resume_config.write_text(
            json.dumps(saved, indent=2),
            encoding='utf-8',
        )
        log.info('Updated resume overrides in %s', resume_config)



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

    parser.add_argument('--run_name', type=str)

    parser.add_argument('--seed', type=int)

    parser.add_argument('--train_dir', type=str)

    parser.add_argument('--train_for_env_steps', type=int)

    parser.add_argument('--training_population', type=int)

    params = parser.parse_args()


    register_custom_components()

    with open(params.config_path, 'r', encoding='utf-8') as f:

        config = yaml.safe_load(f)

    explicit_overrides = set()

    if params.run_name is not None:
        config['name'] = params.run_name
        config.setdefault('global_settings', {})['experiment'] = params.run_name
        explicit_overrides.add('experiment')
    if params.seed is not None:
        config.setdefault('global_settings', {})['seed'] = params.seed
        explicit_overrides.add('seed')
    if params.train_dir is not None:
        config.setdefault('global_settings', {})['train_dir'] = params.train_dir
        explicit_overrides.add('train_dir')
    if params.train_for_env_steps is not None:
        config.setdefault('experiment_settings', {})[
            'train_for_env_steps'
        ] = params.train_for_env_steps
        explicit_overrides.add('train_for_env_steps')

    if params.training_population is not None:
        allowed_populations = {100, 200, 300, 400, 500, 600}
        if params.training_population not in allowed_populations:
            raise ValueError(
                '--training_population must be one of '
                f'{sorted(allowed_populations)}.'
            )
        worker_count = int(config['async_ppo']['num_workers'])
        config.setdefault('environment', {})[
            'training_num_agents_by_worker'
        ] = [params.training_population] * worker_count


    exp, flat_config = validate_config(config)

    if params.training_population is not None:
        flat_config.training_population = params.training_population
        explicit_overrides.add('training_population')

    _sync_resume_cli_overrides(flat_config, explicit_overrides)

    status = run_rl(flat_config)

    return status



if __name__ == '__main__':

    sys.exit(main())
