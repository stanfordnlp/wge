# FormWoB Tasks based on OpenAI world-of-bits.
import time
import os

os.environ['UNIVERSE_NTPDATE_TIMEOUT'] = '1'
os.environ['OPENAI_DOCKER_REPO'] = 'tianlins'

ENV_ID_WHITELIST = [
    'wob.real.Delta-v0'
]

import gym
from gym.envs.registration import register

from universe import wrappers
from universe.spaces.vnc_event import KeyEvent
import universe

from wge.environment import Environment


def register_envs():
    for env_id in ENV_ID_WHITELIST:
        register(
            id=env_id,
            entry_point='universe.wrappers:WrappedVNCEnv',
            max_episode_steps=10**7,
            tags={
                'vnc': True,
                'wob': True,
                'runtime': 'world-of-bits',
            },
        )

try:
    register_envs()
except:
    pass


class FormWoBEnvironment(Environment):
    """FormWoB Environment"""

    def __init__(self, subdomain):
        """ subdomain is the name of the FormWoB task, such as Delta-v0
            we map it directly to universe env_id by adding prefix `wob.real.`
        """
        self.env_id = 'wob.real.' + subdomain
        self.env = gym.make(self.env_id)
        self._num_instances = 0

    def configure(self, num_instances=1, **kwargs):
        self.env.configure(remotes=num_instances, **kwargs)
        self._num_instances = num_instances

    def reset(self, **kwargs):
        self.env.reset()

    def step(self, actions):
        """Take a step in the environment.
        
        Args:
            actions (list[list[VNCEvent]]): a batch of VNCEvent sequences. An empty sequence means no actions.

        Returns:
            states (list[dict]):
                dict['text'] (dict)
                dict['vision'] (np.ndarray)
            rewards (list[float])
            dones (list[bool]): once `done` is True, further actions on that
                instance will give undefined results.
            info (dict): additional debug information.
                Global debug information is directly in the root level
                Local information for instance i is in info['n'][i]
        """
        return self.env.step(actions)

    def close(self):
        self.env.close()

    @property
    def num_instances(self):
        return self._num_instances


if __name__  == '__main__':
    env = Environment.make(domain='formwob', subdomain='Delta-v0')
    env.configure()
    while True:
        observation, _, done, info = env.step([])
        print(observation)
        time.sleep(1)


