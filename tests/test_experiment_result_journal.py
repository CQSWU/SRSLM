from pathlib import Path

import pytest

from run_experiments import (
    _append_journal_record,
    _initialize_result_journal,
    _load_result_journal,
    _journal_task_key,
)


CONTRACT = "a" * 64


def _task(seed=0):
    return {
        "algorithm": "SRSLM-WaitDetectOnly",
        "map_name": "map-a",
        "num_agents": 100,
        "seed": seed,
    }


def test_result_journal_recovers_successes_and_repairs_only_truncated_tail(tmp_path: Path):
    tasks = [_task(0), _task(42)]
    journal = tmp_path / "results.journal.jsonl"
    _initialize_result_journal(journal, CONTRACT, len(tasks))
    first = dict(tasks[0], error=None, avg_throughput=1.0)
    _append_journal_record(
        journal,
        {
            "record_type": "result",
            "task_key": list(_journal_task_key(first)),
            "result": first,
        },
    )
    with journal.open("ab") as stream:
        stream.write(b'{"record_type":"result"')

    recovered = _load_result_journal(
        journal,
        CONTRACT,
        tasks,
        repair_final_record=True,
    )
    assert recovered == [first]
    assert journal.read_bytes().endswith(b"\n")


def test_result_journal_rejects_contract_and_duplicate_success(tmp_path: Path):
    tasks = [_task(0)]
    journal = tmp_path / "results.journal.jsonl"
    _initialize_result_journal(journal, CONTRACT, len(tasks))
    result = dict(tasks[0], error=None)
    record = {
        "record_type": "result",
        "task_key": list(_journal_task_key(result)),
        "result": result,
    }
    _append_journal_record(journal, record)
    _append_journal_record(journal, record)
    with pytest.raises(ValueError, match="repeats successful"):
        _load_result_journal(journal, CONTRACT, tasks)
    with pytest.raises(ValueError, match="contract/header differs"):
        _load_result_journal(journal, "b" * 64, tasks)

