import argparse

import json

import shutil

import sys

import threading

import time

from pathlib import Path


import yaml


import train



PROJECT_ROOT = Path(__file__).resolve().parent



def _validate_reset_target(path: Path) -> Path:

    """Return a resolved, dedicated weights run directory safe to delete.

    Reset is intentionally restricted to a strict descendant of a directory
    named ``weights``.  Broad roots and ancestors of the checkout/home
    directory are rejected even if a configuration is malformed.
    """

    target = Path(path).expanduser().resolve()
    filesystem_root = Path(target.anchor).resolve()
    home = Path.home().resolve()
    weights_root = (PROJECT_ROOT / 'weights').resolve()

    protected = {
        filesystem_root,
        home,
        PROJECT_ROOT,
        weights_root,
    }
    protected.update(PROJECT_ROOT.parents)
    protected.update(home.parents)
    if target in protected:

        raise ValueError(f'Refusing to reset broad or protected path: {target}')

    if not any(parent.name.lower() == 'weights' for parent in target.parents):

        raise ValueError(
            'CAAR reset target must be a dedicated run below a weights '
            f'directory: {target}'
        )

    if target.exists() and not target.is_dir():

        raise ValueError(f'CAAR reset target is not a directory: {target}')

    return target



def _load_train_dir(config_path: Path) -> Path:

    with open(config_path, 'r') as f:

        config = yaml.safe_load(f)


    train_dir = config.get('global_settings', {}).get(
        'train_dir',
        'weights/CAAR',
    )

    return (config_path.parent.parent / train_dir).resolve()



def _find_latest_sf_log(train_dir: Path) -> Path | None:

    candidates = sorted(train_dir.rglob('sf_log.txt'), key=lambda path: path.stat().st_mtime)

    return candidates[-1] if candidates else None



def _find_resume_config(train_dir: Path) -> Path | None:

    candidates = sorted(train_dir.rglob('config.json'), key=lambda path: path.stat().st_mtime)

    return candidates[-1] if candidates else None



def _sync_resume_runtime_config(config_path: Path, train_dir: Path):

    resume_config = _find_resume_config(train_dir)

    if resume_config is None:

        return


    with open(config_path, 'r') as f:

        yaml_config = yaml.safe_load(f)

    with open(resume_config, 'r') as f:

        current_config = json.load(f)


    async_cfg = yaml_config.get('async_ppo', {})

    exp_cfg = yaml_config.get('experiment_settings', {})

    eval_cfg = yaml_config.get('evaluation', {})


    updates = {

        ('num_workers',): async_cfg.get('num_workers'),

        ('num_envs_per_worker',): async_cfg.get('num_envs_per_worker'),

        ('max_policy_lag',): async_cfg.get('max_policy_lag'),

        ('save_best_metric',): exp_cfg.get('save_best_metric'),

        ('save_best_every_sec',): exp_cfg.get('save_best_every_sec'),

        ('save_best_after',): exp_cfg.get('save_best_after'),

        ('train_for_env_steps',): exp_cfg.get('train_for_env_steps'),

        ('env_frameskip',): eval_cfg.get('env_frameskip'),

        ('full_config', 'async_ppo', 'num_workers'): async_cfg.get('num_workers'),

        ('full_config', 'async_ppo', 'num_envs_per_worker'): async_cfg.get('num_envs_per_worker'),

        ('full_config', 'async_ppo', 'max_policy_lag'): async_cfg.get('max_policy_lag'),

        ('full_config', 'experiment_settings', 'save_best_metric'): exp_cfg.get('save_best_metric'),

        ('full_config', 'experiment_settings', 'save_best_every_sec'): exp_cfg.get('save_best_every_sec'),

        ('full_config', 'experiment_settings', 'save_best_after'): exp_cfg.get('save_best_after'),

        ('full_config', 'experiment_settings', 'train_for_env_steps'): exp_cfg.get(
            'train_for_env_steps'
        ),

        ('full_config', 'evaluation', 'env_frameskip'): eval_cfg.get('env_frameskip'),

    }


    applied = []

    for path, value in updates.items():

        if value is None:

            continue

        target = current_config

        for key in path[:-1]:

            if key not in target or not isinstance(target[key], dict):

                target[key] = {}

            target = target[key]

        if target.get(path[-1]) != value:

            target[path[-1]] = value

            applied.append((".".join(path), value))


    if applied:

        with open(resume_config, 'w') as f:

            json.dump(current_config, f, indent=2)

        print(f'Synced resume config at {resume_config}', flush=True)

        for key, value in applied:

            print(f'  {key} -> {value}', flush=True)



def _watch_metrics(train_dir: Path, stop_event: threading.Event, start_from_end: bool):

    watched_file = None

    file_handle = None

    last_position = 0

    metric_tokens = (
        'avg_throughput',
        'stall_rate',
        'revisit_rate',
        'mean_tau_on_chosen_cell',
    )


    try:

        while not stop_event.is_set():

            latest_log = _find_latest_sf_log(train_dir) if train_dir.exists() else None

            if latest_log is None:

                time.sleep(1.0)

                continue


            if watched_file != latest_log:

                if file_handle is not None:

                    file_handle.close()

                watched_file = latest_log

                file_handle = open(watched_file, 'r')

                if start_from_end:

                    file_handle.seek(0, 2)

                    last_position = file_handle.tell()

                else:

                    last_position = 0

                print(f'Watching metrics from {watched_file}', flush=True)


            file_handle.seek(last_position)

            new_lines = file_handle.readlines()

            last_position = file_handle.tell()


            for line in new_lines:

                if any(token in line for token in metric_tokens):

                    print(line.rstrip(), flush=True)


            time.sleep(1.0)

    finally:

        if file_handle is not None:

            file_handle.close()



def main():

    parser = argparse.ArgumentParser(description='Train CAAR with optional reset.')

    parser.add_argument(

        '--config_path',

        type=str,

        help='path to yaml file with CAAR training configuration',

        required=True,

    )

    parser.add_argument(

        '--reset',

        action='store_true',

        help='delete the configured CAAR train_dir before starting training',

    )

    parser.add_argument(

        '--no_watch',

        action='store_true',

        help='disable realtime throughput/ACO metric printing from sf_log.txt',

    )

    args = parser.parse_args()


    config_path = Path(args.config_path).resolve()

    train_dir = _load_train_dir(config_path)

    if args.reset:

        train_dir = _validate_reset_target(train_dir)

        if train_dir.exists():

            print(f'Removing existing CAAR weights at {train_dir}', flush=True)

            shutil.rmtree(train_dir)

        else:

            print(f'No existing CAAR weights found at {train_dir}', flush=True)

    else:

        _sync_resume_runtime_config(config_path, train_dir)


    sys.argv = [sys.argv[0], '--config_path', str(config_path)]


    stop_event = threading.Event()

    watcher = None

    if not args.no_watch:

        watcher = threading.Thread(

            target=_watch_metrics,

            args=(train_dir, stop_event, not args.reset),

            daemon=True,

        )

        watcher.start()


    try:

        return train.main()

    finally:

        stop_event.set()

        if watcher is not None:

            watcher.join(timeout=2.0)



if __name__ == '__main__':

    sys.exit(main())
