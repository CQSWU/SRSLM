import torch

from agents.direct import Direct, DirectConfig


def test_direct_public_names_are_current():
    assert DirectConfig().name == "Direct"
    assert Direct.__mro__[1].__name__ == "NoReweight"


def test_direct_uses_capped_relu_pressure():
    pressure = torch.tensor([[-1.0, 0.0, 0.5, 2.0, 3.0]])
    result = Direct.transform_pressures(pressure)
    assert torch.equal(result, torch.tensor([[0.0, 0.0, 0.5, 2.0, 2.0]]))


def test_direct_subtracts_pressure_from_logits():
    instance = object.__new__(Direct)
    instance.pressure_scale = 1.0
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    pressure = torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0]])
    assert torch.equal(instance.adjust_logits(logits, pressure), logits - pressure)

