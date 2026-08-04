import multiprocessing
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Extra, Field, root_validator, validator

from pomapf_env.pomapf_config import POMAPFConfig

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


    encoder_custom: Optional[str] = None

    encoder_subtype: str = 'resnet_impala'

    encoder_extra_fc_layers: int = 1


    caar_num_filters: int = 64

    caar_num_res_blocks: int = 3

    caar_tau_num_filters: int = 8

    caar_tau_num_conv_layers: int = 1

    caar_tau_num_res_blocks: int = 0

    caar_tau_hidden_size: int = 0

    caar_learn_residual: bool = True

    caar_contextual_pressure: bool = False

    hidden_size: int = 512

    nonlinearity: str = 'relu'

    policy_initialization: str = 'orthogonal'

    policy_init_gain: float = 1.0

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

    name: Literal[
        "POMAPF-v0",
        "POMAPF-ST-v0",
    ] = "POMAPF-v0"

    tau_rho: float = Field(0.1, gt=0.0, le=1.0)

    tau_radius: Optional[int] = Field(None, ge=1)





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
    def validate_caar_settings(cls, values):
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
        if settings is None or settings.encoder_custom != 'caar':
            return values

        normalized_keys = settings.normalize_input_keys
        if (
            settings.normalize_input
            and normalized_keys
            and 'tau' in normalized_keys
        ):
            raise ValueError(
                'CAAR tau must not be running-normalized because its signed '
                'pressure values are applied directly to action logits.'
            )
        if settings.caar_contextual_pressure and not settings.caar_learn_residual:
            raise ValueError(
                'caar_contextual_pressure requires caar_learn_residual=true.'
            )
        if (
            environment is not None
            and environment.name == 'POMAPF-ST-v0'
            and settings.caar_learn_residual
            and not settings.caar_contextual_pressure
        ):
            raise ValueError(
                'Legacy CAAR residual mode is retired; use the current CAAR '
                'configuration (both flags true).'
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
