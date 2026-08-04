from typing import Literal


from pogema import GridConfig



class POMAPFConfig(GridConfig):

    integration: Literal['SampleFactory'] = 'SampleFactory'

    collision_system: Literal['block_both', 'soft'] = 'soft'

    observation_type: Literal['POMAPF', 'MAPF'] = 'POMAPF'
