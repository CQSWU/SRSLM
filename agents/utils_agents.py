from typing import Optional

from pydantic import BaseModel

SUPPORTED_COLLISION_SYSTEMS = ("block_both", "soft", "priority")


class AlgoBase(BaseModel):
    name: str = None
    device: str = "mps"
    seed: Optional[int] = 0


class ResultsHolder:
    def __init__(self):
        self.results = {}

    def after_step(self, infos):
        if "metrics" in infos[0]:
            self.results.update(infos[0]["metrics"])

    def get_final(self):
        return self.results
