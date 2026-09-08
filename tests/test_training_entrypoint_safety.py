"""Current training has no reset wrapper and only resumes the explicit run."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import train


ROOT = Path(__file__).resolve().parents[1]


def _run_config(path, frames=50_000_000):
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        'train_for_env_steps': frames,
        'full_config': {'experiment_settings': {'train_for_env_steps': frames}},
    }) + '\n', encoding='utf-8')
    return path.read_bytes()


def test_retired_reset_wrapper_is_not_shipped():
    assert not (ROOT / 'train_caar.py').exists()
    assert '"train_caar"' not in (ROOT / 'pyproject.toml').read_text(encoding='utf-8')


def test_current_entry_rejects_reset_before_registering_or_writing():
    args = ['train.py', '--config_path', 'unused.yaml', '--reset']
    with patch('sys.argv', args), patch.object(train, 'register_custom_components') as register, \
            patch.object(train, '_sync_resume_cli_overrides') as sync, \
            patch.object(train, 'run_rl') as run:
        with pytest.raises(SystemExit) as raised:
            train.main()
        assert raised.value.code == 2
        register.assert_not_called()
        sync.assert_not_called()
        run.assert_not_called()


def test_no_explicit_override_leaves_original_config_untouched(tmp_path):
    path = tmp_path / 'selected-run' / 'config.json'
    original = _run_config(path)
    original_mtime = path.stat().st_mtime_ns
    cfg = SimpleNamespace(train_dir=str(tmp_path), experiment='selected-run',
                          train_for_env_steps=100_000_000, cli_args={'seed': 42})
    train._sync_resume_cli_overrides(cfg, set())
    assert path.read_bytes() == original
    assert path.stat().st_mtime_ns == original_mtime
    assert cfg.cli_args == {'seed': 42}


def test_explicit_resume_updates_only_named_run_not_newest_config(tmp_path):
    selected = tmp_path / 'selected-run' / 'config.json'
    unrelated = tmp_path / 'newer-unrelated-run' / 'config.json'
    _run_config(selected)
    other_original = _run_config(unrelated, frames=7)
    os.utime(selected, ns=(1_000_000_000, 1_000_000_000))
    os.utime(unrelated, ns=(2_000_000_000, 2_000_000_000))
    assert unrelated.stat().st_mtime_ns > selected.stat().st_mtime_ns
    cfg = SimpleNamespace(train_dir=str(tmp_path), experiment='selected-run',
                          train_for_env_steps=100_000_000, cli_args={})
    train._sync_resume_cli_overrides(cfg, {'train_for_env_steps'})
    saved = json.loads(selected.read_text(encoding='utf-8'))
    assert saved['train_for_env_steps'] == 100_000_000
    assert saved['full_config']['experiment_settings']['train_for_env_steps'] == 100_000_000
    assert cfg.cli_args['train_for_env_steps'] == 100_000_000
    assert unrelated.read_bytes() == other_original
