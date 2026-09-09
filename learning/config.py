import multiprocessing
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, Extra, Field, root_validator, validator

from pomapf_env.pomapf_config import POMAPFConfig

# Both protocols are explicitly supported; saved configs still select one.
AUDITED_COLLISION_SYSTEMS = ('block_both', 'soft')

# Immutable checkpoint JSONs contain these former defaults. Drop them only
# when reading saved checkpoints; new YAMLs still reject unused fields.
OBSOLETE_SAVED_SETTINGS = frozenset({
    'trace_residual_base_weights_path', 'trace_residual_filters',
    'trace_residual_hidden', 'trace_residual_use_agents',
    'trace_residual_use_base_logits', 'trace_residual_gate',
    'trace_residual_gate_threshold', 'trace_residual_gate_temperature',
    'trace_residual_gate_rate', 'trace_spatial_input_contract',
    'trace_spatial_hidden_dim', 'trace_spatial_trace_view',
    'epom_trace_num_filters', 'epom_trace_num_res_blocks',
    'epom_trace_embedding_size', 'epom_trace_epom_feature_size',
    'epom_trace_fusion_size', 'epom_trace_head_size',
    'trace_context_filters', 'trace_context_embedding_size',
    'trace_context_hidden_projection', 'trace_context_fusion_size',
    'trace_context_head_size', 'trace_context_residual_cap',
    # Serialized defaults from the retired caar+tau actor. They are inert in
    # the retained NoReweight, EPOM-L, ARPE and Switcher checkpoints.
    'caar_tau_num_filters', 'caar_tau_num_conv_layers',
    'caar_tau_num_res_blocks', 'caar_tau_hidden_size',
    'caar_learn_residual', 'caar_contextual_pressure',
    'caar_pressure_head_mode', 'caar_pressure_output_transform',
    'caar_pressure_cap', 'caar_pressure_init', 'caar_reweight_wait_action',
    'trace_correction_mode',
})


def checkpoint_experiment_config(config):
    """Read retained checkpoints without modifying original config files."""
    normalized = deepcopy(config)
    settings = normalized.get('experiment_settings', {})
    for key in OBSOLETE_SAVED_SETTINGS:
        settings.pop(key, None)
    if settings.get('encoder_custom') in {
        'pogema_residual', 'epom_finetune', 'caar', 'epom_trace_context',
    }:
        # An unused Switcher default leaked into early base-policy configs.
        # Do not migrate actual Switcher configs: there it can change routing.
        normalized.get('environment', {}).pop('switcher_rule_guard_enabled', None)
    return normalized

class AsyncPPO(BaseModel, extra=Extra.forbid):

    async_rl: bool = True

    experiment_summaries_interval: int = 20

    adam_eps: float = 1e-6

    adam_beta1: float = 0.9

    adam_beta2: float = 0.999

    gae_lambda: float = 0.95

    rollout: int = 32

    num_workers: int = multiprocessing.cpu_count()

    recurrence: int = 32

    use_rnn: bool = True

    rnn_type: str = 'gru'

    rnn_num_layers: int = 1

    ppo_clip_ratio: float = 0.1

    ppo_clip_value: float = 1.0

    batch_size: int = 1024

    num_batches_per_iteration: int = 1

    num_epochs: int = 1

    max_grad_norm: float = 4.0


    exploration_loss_coeff: float = 0.003

    value_loss_coeff: float = 0.5

    kl_loss_coeff: float = 0.0

    exploration_loss: str = 'entropy'

    num_envs_per_worker: int = 2

    worker_num_splits: int = 2

    num_policies: int = 1

    policy_workers_per_policy: int = 1

    max_policy_lag: int = 10000

    decorrelate_experience_max_seconds: int = 10

    decorrelate_envs_on_one_worker: bool = True


    with_vtrace: bool = True

    vtrace_rho: float = 1.0

    vtrace_c: float = 1.0

    set_workers_cpu_affinity: bool = True

    force_envs_single_thread: bool = True

    default_niceness: int = 0

    actor_worker_gpus: List[int] = Field(default_factory=list)


    with_pbt: bool = False

    pbt_optimize_gamma: bool = True

    pbt_mix_policies_in_one_env: bool = True

    pbt_period_env_steps: int = 3_000_000

    pbt_start_mutation: int = 20_000_000

    pbt_replace_fraction: float = 0.3

    pbt_mutation_rate: float = 0.15

    pbt_replace_reward_gap: float = 0.05

    pbt_replace_reward_gap_absolute: float = 1e-6

    pbt_target_objective: str = 'true_reward'

    benchmark: bool = False



class ExperimentSettings(BaseModel, extra=Extra.forbid):

    save_every_sec: int = 120

    save_best_every_sec: int = 5

    save_best_after: int = 100000

    save_best_metric: str = 'reward'

    keep_checkpoints: int = 1

    save_milestones_sec: int = -1

    stats_avg: int = 100

    learning_rate: float = 1e-4

    train_for_env_steps: int = 10_000_000_000

    train_for_seconds: int = 10_000_000_000


    obs_subtract_mean: float = 0.0

    obs_scale: float = 1.0

    normalize_input: bool = True

    normalize_input_keys: Optional[List[str]] = None


    gamma: float = 0.99

    reward_scale: float = 1.0

    reward_clip: float = 10.0


    encoder_custom: Optional[Literal[
        'pogema_residual', 'epom_finetune', 'caar', 'epom_trace_context',
        'switcher', 'switcher_all_state',
    ]] = None

    encoder_subtype: str = 'resnet_impala'

    encoder_extra_fc_layers: int = 1

    encoder_mlp_layers: List[int] = Field(default_factory=lambda: [512, 512])

    decoder_mlp_layers: List[int] = Field(default_factory=list)


    caar_num_filters: int = 64

    caar_num_res_blocks: int = 3

    pogema_encoder_num_filters: int = Field(64, ge=1)

    pogema_encoder_num_res_blocks: int = Field(3, ge=0)

    epom_base_weights_path: str = 'weights/EPOM/EPOM'

    # Direct baseline and ARPE entropy threshold.
    trace_rule_scale: float = Field(1.0, ge=0.0)
    trace_gate_threshold: float = Field(0.46371241)

    # "context" is inert metadata in saved non-Trace checkpoints; it cannot
    # instantiate a trace model. The only retained trainable branch is below.
    trace_context_architecture: Literal[
        'context',
        'paper_entropy_fusion',
    ] = 'context'

    trace_context_learned_gate: Literal[
        'entropy',
        'all',
    ] = 'entropy'

    hidden_size: int = 512

    nonlinearity: str = 'relu'

    policy_initialization: str = 'orthogonal'

    policy_init_gain: float = 1.0

    switcher_initial_ao_probability: float = Field(0.1, gt=0.0, lt=1.0)

    actor_critic_share_weights: bool = True


    adaptive_stddev: bool = True

    initial_stddev: float = 1.0


    lr_schedule: str = 'kl_adaptive_minibatch'

    lr_schedule_kl_threshold: Optional[float] = None



class GlobalSettings(BaseModel, extra=Extra.forbid):

    algo: str = 'APPO'

    env: Optional[str] = None

    experiment: Optional[str] = None

    train_dir: str = 'weights/train_dir'

    device: str = 'gpu'

    serial_mode: bool = False

    seed: Optional[int] = None

    cli_args: Dict[str, Any] = Field(default_factory=dict)

    with_wandb: bool = False



class Evaluation(BaseModel, extra=Extra.forbid):

    fps: int = 0

    no_render: bool = True

    policy_index: int = 0

    env_frameskip: Optional[int] = None



class Environment(BaseModel, extra=Extra.forbid):

    grid_config: POMAPFConfig = Field(default_factory=POMAPFConfig)

    # Optional training-only population assignment. Sample Factory creates
    # multiple vector environments per rollout worker; assigning by worker
    # keeps every environment in one worker at the same population.
    training_num_agents_by_worker: Optional[List[int]] = Field(
        None,
        min_items=1,
    )

    name: Literal[
        "POMAPF-v0",
        "POMAPF-EPOM-v0",
        "POMAPF-EPOM-ST-v0",
        "POMAPF-SRSLM-v0",
        "POMAPF-SRSLM-NoWait-v0",
    ] = "POMAPF-v0"

    tau_rho: float = Field(0.1, gt=0.0, le=1.0)
    # NOTE ON CONVENTION: tau_rho is the EVAPORATION rate.  AcoState sets
    # decay = 1 - tau_rho, so tau_rho=0.1 means a retention factor of 0.9 and a
    # memory of roughly 22 steps.  Papers that write P_t = rho*P_{t-1} + O_t
    # use rho for RETENTION; writing that rho here would invert the memory.
    trace_variant: str = Field('real')
    tau_raw: bool = Field(False)

    tau_radius: Optional[int] = Field(None, ge=1)

    grid_memory_obs_radius: int = Field(7, ge=1)

    switcher_caar_weights_path: str = (
        "weights/CAAR-p-identity-r5-1b/CAAR-P-Identity-R5-1B"
    )

    switcher_caar_checkpoint_kind: Literal[
        "auto", "latest", "best"
    ] = "latest"

    switcher_caar_device: str = "auto"

    switcher_max_planning_steps: int = Field(10_000, gt=0)

    switcher_team_reward_coefficient: float = 1.0

    trace_context_team_reward_coefficient: float = 1.0

    switcher_feature_schema: str = "srslm_switcher_state_v3"





class Experiment(BaseModel, extra=Extra.forbid):

    name: Optional[str] = None

    environment: Environment = Field(default_factory=Environment)

    async_ppo: AsyncPPO = Field(default_factory=AsyncPPO)

    experiment_settings: ExperimentSettings = Field(
        default_factory=ExperimentSettings
    )

    global_settings: GlobalSettings = Field(default_factory=GlobalSettings)

    evaluation: Evaluation = Field(default_factory=Evaluation)

    @root_validator
    def validate_arpe_settings(cls, values):
        environment = values.get('environment')
        global_settings = values.get('global_settings')
        if (
            environment is not None
            and global_settings is not None
            and global_settings.env != environment.name
        ):
            raise ValueError(
                'global_settings.env must match environment.name, got '
                f'{global_settings.env!r} and {environment.name!r}.'
            )
        if global_settings is not None and not global_settings.experiment:
            raise ValueError('An experiment name is required.')

        settings = values.get('experiment_settings')
        if settings is None:
            return values

        if settings.encoder_custom == 'epom_finetune':
            if environment is None or environment.name != 'POMAPF-EPOM-v0':
                raise ValueError(
                    "EPOM fine-tuning requires "
                    "environment.name='POMAPF-EPOM-v0'."
                )
            grid = environment.grid_config
            if environment.grid_memory_obs_radius != 7:
                raise ValueError(
                    'Official EPOM requires grid_memory_obs_radius=7.'
                )
            if grid.obs_radius != 5:
                raise ValueError('EPOM fine-tuning requires obs_radius=5.')
            if grid.max_episode_steps != 512:
                raise ValueError(
                    'EPOM fine-tuning requires max_episode_steps=512.'
                )
            if grid.num_agents != 200:
                raise ValueError('EPOM fine-tuning requires num_agents=200.')
            if grid.on_target != 'restart':
                raise ValueError(
                    "EPOM fine-tuning requires on_target='restart'."
                )
            if grid.collision_system not in AUDITED_COLLISION_SYSTEMS:
                raise ValueError(
                    'EPOM fine-tuning runs only under an audited execution '
                    f'model; received {grid.collision_system!r}, audited '
                    f'{AUDITED_COLLISION_SYSTEMS}.'
                )
            if str(grid.map_name).replace('\\', '/') != 'maps/train.yaml':
                raise ValueError(
                    "EPOM fine-tuning requires map_name='maps/train.yaml'."
                )
            if settings.normalize_input:
                raise ValueError(
                    'EPOM fine-tuning must set normalize_input=false because '
                    'official EPOM v0 has no observation-normalizer state.'
                )
            if settings.hidden_size != 512:
                raise ValueError('Official EPOM requires hidden_size=512.')
            if settings.pogema_encoder_num_filters != 64:
                raise ValueError('Official EPOM requires 64 encoder filters.')
            if settings.pogema_encoder_num_res_blocks != 3:
                raise ValueError(
                    'Official EPOM requires 3 encoder residual blocks.'
                )
            if settings.encoder_extra_fc_layers != 1:
                raise ValueError('Official EPOM requires one encoder FC layer.')
            async_ppo = values.get('async_ppo')
            if async_ppo is None or not async_ppo.use_rnn:
                raise ValueError('Official EPOM requires its recurrent policy.')
            if async_ppo.rnn_type != 'gru' or async_ppo.rnn_num_layers != 1:
                raise ValueError('Official EPOM requires a single-layer GRU.')
            if async_ppo.recurrence != 32 or async_ppo.rollout != 32:
                raise ValueError(
                    'EPOM fine-tuning keeps the official rollout/recurrence of 32.'
                )
            official_ppo = {
                'batch_size': (async_ppo.batch_size, 4096),
                'num_envs_per_worker': (async_ppo.num_envs_per_worker, 2),
                'num_epochs': (async_ppo.num_epochs, 1),
                'max_grad_norm': (async_ppo.max_grad_norm, 5.0),
                'with_vtrace': (async_ppo.with_vtrace, False),
                'max_policy_lag': (async_ppo.max_policy_lag, 100),
                'exploration_loss_coeff': (
                    async_ppo.exploration_loss_coeff,
                    0.01,
                ),
                'value_loss_coeff': (async_ppo.value_loss_coeff, 0.5),
                'ppo_clip_ratio': (async_ppo.ppo_clip_ratio, 0.1),
                'gae_lambda': (async_ppo.gae_lambda, 0.95),
                'adam_eps': (async_ppo.adam_eps, 1e-6),
                'adam_beta1': (async_ppo.adam_beta1, 0.9),
                'adam_beta2': (async_ppo.adam_beta2, 0.999),
            }
            mismatched_ppo = {
                key: {'fine_tune': actual, 'official': expected}
                for key, (actual, expected) in official_ppo.items()
                if actual != expected
            }
            if mismatched_ppo:
                raise ValueError(
                    'EPOM fine-tuning PPO settings differ from official v0: '
                    f'{mismatched_ppo}'
                )
            if async_ppo.num_workers > 12:
                raise ValueError(
                    'EPOM fine-tuning exceeds the Server2 cap of 12 workers.'
                )
            if settings.learning_rate != 1e-4:
                raise ValueError(
                    'EPOM fine-tuning keeps the official learning_rate=1e-4.'
                )
            if settings.gamma != 0.99:
                raise ValueError('EPOM fine-tuning keeps the official gamma=0.99.')
            if settings.lr_schedule != 'constant':
                raise ValueError('EPOM fine-tuning requires a constant LR schedule.')
            return values

        if settings.encoder_custom == 'epom_trace_context':
            if environment is None or environment.name != 'POMAPF-EPOM-ST-v0':
                raise ValueError(
                    "EPOM trace-context training requires "
                    "environment.name='POMAPF-EPOM-ST-v0'."
                )
            grid = environment.grid_config
            if environment.grid_memory_obs_radius != 7:
                raise ValueError(
                    'EPOM trace-context training requires '
                    'grid_memory_obs_radius=7.'
                )
            if environment.tau_radius != 5:
                raise ValueError(
                    'Paper EPOM trace-context training requires tau_radius=5 '
                    '(an 11x11 trace crop).'
                )
            if settings.trace_context_architecture != 'paper_entropy_fusion':
                raise ValueError('Only the paper_entropy_fusion ARPE architecture is retained.')
            if environment.tau_raw is not False:
                raise ValueError(
                    'EPOM trace-context architecture '
                    f'{settings.trace_context_architecture!r} requires '
                    'tau_raw=false.'
                )
            if environment.trace_variant != 'real':
                raise ValueError(
                    "EPOM trace-context training requires trace_variant='real'."
                )
            if not np.isfinite(
                environment.trace_context_team_reward_coefficient
            ):
                raise ValueError(
                    'trace_context_team_reward_coefficient must be finite.'
                )
            if grid.obs_radius != 5:
                raise ValueError(
                    'EPOM trace-context training requires obs_radius=5.'
                )
            if grid.max_episode_steps != 512:
                raise ValueError(
                    'EPOM trace-context training requires max_episode_steps=512.'
                )
            if grid.num_agents != 200:
                raise ValueError(
                    'EPOM trace-context training requires baseline '
                    'num_agents=200.'
                )
            if grid.on_target != 'restart':
                raise ValueError(
                    "EPOM trace-context training requires on_target='restart'."
                )
            if grid.collision_system not in AUDITED_COLLISION_SYSTEMS:
                raise ValueError(
                    'EPOM trace-context training runs only under an audited '
                    f'execution model; received {grid.collision_system!r}, '
                    f'audited {AUDITED_COLLISION_SYSTEMS}.'
                )
            if settings.trace_gate_threshold != 0.46371241:
                raise ValueError('Paper ARPE fixes trace_gate_threshold=0.46371241.')
            if settings.trace_rule_scale != 1.0:
                raise ValueError('Paper ARPE fixes trace_rule_scale=1.0.')
            expected_map_name = 'maps/train_capacity_n600.yaml'
            if str(grid.map_name).replace('\\', '/') != expected_map_name:
                raise ValueError(
                    'EPOM trace-context architecture '
                    f'{settings.trace_context_architecture!r} requires '
                    f'map_name={expected_map_name!r}.'
                )
            if settings.normalize_input:
                raise ValueError(
                    'EPOM trace-context training requires normalize_input=false '
                    'to preserve the frozen EPOM-L input contract.'
                )
            if settings.hidden_size != 512:
                raise ValueError('Official EPOM requires hidden_size=512.')
            if settings.pogema_encoder_num_filters != 64:
                raise ValueError('Official EPOM requires 64 encoder filters.')
            if settings.pogema_encoder_num_res_blocks != 3:
                raise ValueError(
                    'Official EPOM requires 3 encoder residual blocks.'
                )
            if settings.encoder_extra_fc_layers != 1:
                raise ValueError('Official EPOM requires one encoder FC layer.')
            if not settings.epom_base_weights_path:
                raise ValueError(
                    'EPOM trace-context training requires EPOM-L base weights.'
                )
            async_ppo = values.get('async_ppo')
            if async_ppo is None or not async_ppo.use_rnn:
                raise ValueError('Official EPOM requires its recurrent policy.')
            training_populations = (
                environment.training_num_agents_by_worker
            )
            if training_populations is not None:
                allowed_populations = {100, 200, 300, 400, 500, 600}
                invalid_populations = sorted(
                    set(training_populations) - allowed_populations
                )
                if invalid_populations:
                    raise ValueError(
                        'EPOM trace-context training_num_agents_by_worker '
                        'values must come from '
                        '{100, 200, 300, 400, 500, 600}; got '
                        f'{invalid_populations}.'
                    )
                if len(training_populations) != async_ppo.num_workers:
                    raise ValueError(
                        'EPOM trace-context '
                        'training_num_agents_by_worker length must equal '
                        'async_ppo.num_workers; got '
                        f'{len(training_populations)} and '
                        f'{async_ppo.num_workers}.'
                    )
            if async_ppo.rnn_type != 'gru' or async_ppo.rnn_num_layers != 1:
                raise ValueError('Official EPOM requires a single-layer GRU.')
            if async_ppo.recurrence != 32 or async_ppo.rollout != 32:
                raise ValueError(
                    'EPOM trace-context training keeps rollout/recurrence=32.'
                )
            epom_l_ppo = {
                'batch_size': (async_ppo.batch_size, 4096),
                'num_envs_per_worker': (async_ppo.num_envs_per_worker, 2),
                'num_epochs': (async_ppo.num_epochs, 1),
                'max_grad_norm': (async_ppo.max_grad_norm, 5.0),
                'with_vtrace': (async_ppo.with_vtrace, False),
                'max_policy_lag': (async_ppo.max_policy_lag, 100),
                'exploration_loss_coeff': (
                    async_ppo.exploration_loss_coeff,
                    0.01,
                ),
                'value_loss_coeff': (async_ppo.value_loss_coeff, 0.5),
                'ppo_clip_ratio': (async_ppo.ppo_clip_ratio, 0.1),
                'gae_lambda': (async_ppo.gae_lambda, 0.95),
                'adam_eps': (async_ppo.adam_eps, 1e-6),
                'adam_beta1': (async_ppo.adam_beta1, 0.9),
                'adam_beta2': (async_ppo.adam_beta2, 0.999),
            }
            mismatched_ppo = {
                key: {'trace_context': actual, 'epom_l': expected}
                for key, (actual, expected) in epom_l_ppo.items()
                if actual != expected
            }
            if mismatched_ppo:
                raise ValueError(
                    'EPOM trace-context PPO settings differ from EPOM-L: '
                    f'{mismatched_ppo}'
                )
            if async_ppo.num_workers > 12:
                raise ValueError(
                    'EPOM trace-context training exceeds the 12-worker cap.'
                )
            if settings.learning_rate != 1e-4:
                raise ValueError(
                    'EPOM trace-context training requires learning_rate=1e-4.'
                )
            if settings.gamma != 0.99:
                raise ValueError(
                    'EPOM trace-context training requires gamma=0.99.'
                )
            if settings.lr_schedule != 'constant':
                raise ValueError(
                    'EPOM trace-context training requires a constant LR schedule.'
                )
            return values

        if settings.encoder_custom in ('switcher', 'switcher_all_state'):
            expected_environments = (
                {'POMAPF-SRSLM-v0'}
                if settings.encoder_custom == 'switcher'
                else {'POMAPF-SRSLM-NoWait-v0'}
            )
            if environment is None or environment.name not in expected_environments:
                raise ValueError(
                    "Switcher training requires "
                    "environment.name in "
                    f"{sorted(expected_environments)!r}."
                )
            if environment.grid_config.collision_system not in AUDITED_COLLISION_SYSTEMS:
                raise ValueError(
                    'Switcher training runs only under an audited execution '
                    f'model; received '
                    f'{environment.grid_config.collision_system!r}, audited '
                    f'{AUDITED_COLLISION_SYSTEMS}.'
                )
            if environment.switcher_feature_schema != 'srslm_switcher_state_v3':
                raise ValueError('Unsupported Switcher state schema.')
            if not environment.switcher_caar_weights_path:
                raise ValueError('Switcher training requires frozen ARPE weights.')
            if settings.normalize_input:
                raise ValueError(
                    'Switcher requires normalize_input=false so candidate '
                    'one-hot actions remain exact.'
                )
            if not np.isfinite(environment.switcher_team_reward_coefficient):
                raise ValueError(
                    'switcher_team_reward_coefficient must be finite.'
                )
            async_ppo = values.get('async_ppo')
            if async_ppo is not None and async_ppo.use_rnn:
                raise ValueError(
                    'Switcher is a feed-forward local-state policy; '
                    'set use_rnn=false.'
                )
            if async_ppo is not None and async_ppo.with_vtrace:
                raise ValueError(
                    'Switcher actor masking requires with_vtrace=false.'
                )
            if settings.encoder_custom == 'switcher':
                from learning.switcher_learner_patch import (
                    patch_switcher_learner_losses,
                )

                patch_switcher_learner_losses()
            return values

        if settings.encoder_custom != 'caar':
            return values

        if environment is None or environment.name != 'POMAPF-v0':
            raise ValueError(
                "The retained caar encoder is NoReweight and requires "
                "environment.name='POMAPF-v0' without tau."
            )
        return values


    @validator('global_settings')

    def seed_initialization(cls, v, values):

        environment = values.get('environment')
        if v.env is None and environment is not None:

            v.env = environment.name

        if v.experiment is None:

            v.experiment = values['name']

        return v
