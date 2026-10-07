import pytest
import torch

from agents.common import resolve_device


@pytest.mark.parametrize(
    "requested,cuda_available,mps_available,expected",
    [
        ("cpu", False, False, "cpu"),
        ("cpu", True, True, "cpu"),
        ("cuda", True, False, "cuda"),
        ("cuda:2", True, True, "cuda:2"),
        ("cuda:2", False, True, "mps"),
        ("cuda:2", False, False, "cpu"),
        ("auto", True, True, "cuda"),
        ("auto", False, True, "mps"),
        ("auto", False, False, "cpu"),
        ("mps", True, True, "cuda"),
        ("mps", False, True, "mps"),
        ("mps", False, False, "cpu"),
        ("CPU", True, False, "cuda"),
    ],
)
def test_device_resolution_preserves_explicit_cpu_cuda_index_and_fallbacks(
    monkeypatch, requested, cuda_available, mps_available, expected
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps_available)
    assert resolve_device(requested) == torch.device(expected)
