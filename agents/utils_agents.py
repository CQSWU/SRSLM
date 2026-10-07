from typing import Optional

from pydantic import BaseModel

SUPPORTED_COLLISION_SYSTEMS = ("block_both", "soft", "priority")


class AlgoBase(BaseModel):
    name: str = None
    device: str = "mps"
    seed: Optional[int] = 0
