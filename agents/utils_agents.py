import pathlib

from datetime import datetime

from typing import Optional


from pogema.svg_animation.animation_wrapper import AnimationConfig, AnimationMonitor

from pydantic import BaseModel


from pomapf_env.env import make_pomapf

from pomapf_env.pomapf_config import POMAPFConfig



class AlgoBase(BaseModel):

    name: str = None

    device: str = 'mps'

    seed: Optional[int] = 0



def run_algorithm(

    algo,

    map_name='sc1-AcrosstheCape',

    max_episode_steps=512,

    seed=None,

    num_agents=64,

    obs_radius=None,

    animate=False,

    on_target='restart',

    save_results=True,

    collision_system=None,

    map_text=None,

):
    # Keep lightweight planning utilities (including the SVG demo generator)
    # usable without importing the policy-training runtime.
    import torch


    gc_kwargs = dict(

        max_episode_steps=max_episode_steps,

        seed=seed,

        num_agents=num_agents,

        on_target=on_target,

    )

    if obs_radius is not None:

        gc_kwargs['obs_radius'] = obs_radius

    if collision_system is not None:

        gc_kwargs['collision_system'] = collision_system

    if map_text is not None:

        gc_kwargs['map'] = map_text

        gc_kwargs['map_name'] = None

    else:

        gc_kwargs['map_name'] = map_name


    gc = POMAPFConfig(**gc_kwargs)

    algo_name = type(algo).__name__

    env = make_pomapf(grid_config=gc, with_animations=False)

    if animate:

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        map_short = map_name.split('-')[-1] if '-' in map_name else map_name

        anim_dir = str(pathlib.Path('renders') / f'{map_short}_{algo_name}_{num_agents}agents_{timestamp}')

        env = AnimationMonitor(env, AnimationConfig(directory=anim_dir))


    try:

        obs, _ = env.reset()

        algo.after_reset()

        if hasattr(algo, 'set_grid_config'):

            algo.set_grid_config(env.grid_config)

        if hasattr(algo, 'set_env'):

            algo.set_env(env)

        results_holder = ResultsHolder()


        dones = [False for _ in range(len(obs))]

        infos = [{'is_active': True} for _ in range(len(obs))]

        rew = [0 for _ in range(len(obs))]

        with torch.no_grad():

            while True:

                obs, rew, terminated, truncated, infos = env.step(algo.act(obs, rew, dones, infos))

                dones = [t or tr for t, tr in zip(terminated, truncated)]

                results_holder.after_step(infos)

                algo.after_step(dones)


                if all(dones):

                    break


        results = results_holder.get_final()

        results['algorithm'] = algo_name

        return results

    finally:

        env.close()



class ResultsHolder:

    def __init__(self):

        self.results = dict()


    def after_step(self, infos):

        if 'metrics' in infos[0]:

            self.results.update(**infos[0]['metrics'])


    def get_final(self):

        return self.results
