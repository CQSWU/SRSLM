from typing import Optional

from pydantic import BaseModel

SUPPORTED_COLLISION_SYSTEMS = ("block_both", "soft", "priority")


def resolve_device(requested):
    import torch

    if requested == "cpu":
        return torch.device("cpu")
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AlgoBase(BaseModel):
    name: str = None
    device: str = "mps"
    seed: Optional[int] = 0
