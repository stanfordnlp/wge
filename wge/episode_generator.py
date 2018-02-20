import logging
import random
import sys
import traceback
from abc import abstractmethod, ABCMeta
from collections import namedtuple

from gtd.chrono import verboserate
from gtd.log import indent
from wge.miniwob.action import MiniWoBTerminate
from wge.rl import Episode, Experience, BaseExperience, Trace, Justification


class EpisodeGenerator(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, policy, test):
        """List of Episodes.
        
        Args:
            policy (Policy): policy used to roll out Episode
            test (bool): a flag passed to policy, indicating whether to use train-time or test-time policy

        Returns:
            episodes (list[Episode])
        """
        # TODO: Change the signature to match BasicEpisodeGenerator.__call__
        raise NotImplementedError

    @staticmethod
    def _get_random_seeds(n):
        """Get a random seed for each episode."""
        return [random.randint(0, sys.maxint) for _ in xrange(n)]


class BasicEpisodeGenerator(EpisodeGenerator):
    MAX_RETRIES = 5

    def __init__(self, env, max_steps_per_episode, visualize_attention_interval):
        self._env = env
        self._max_steps_per_episode = max_steps_per_episode
        self._visualize_attention_interval = visualize_attention_interval
        self._num_retries = 0

    def __call__(self, policy, test_env, test_policy, seeds=None, record_screenshots=False):
        """
        
        Args:
            policy (Policy)
            test_env (bool): whether to run in test environments
            test_policy (bool): whether to run the policy in test-time mode
            seeds (list[int]): random seeds to initialize the environment
            record_screenshots (bool): whether to record screenshots

        Returns:
            list[Episode]
        """
        while self._num_retries <= self.MAX_RETRIES:
            try:
                episodes = self._get_episodes(policy, test_env, test_policy,
                        seeds, record_screenshots)
                self._num_retries = 0       # Reset
                return episodes
            except Exception as e:
                self._num_retries += 1
                logging.error('#' * 60)
                logging.error('=== SOMETHING WRONG HAPPENED !!! === (Retry attempt %d / %d)',
                        self._num_retries, self.MAX_RETRIES)
                traceback.print_exc()
                logging.error('Will restart the environment and try again ...')
                logging.error('#' * 60)
        raise RuntimeError('Envionment died too many times!')

    def _get_episodes(self, policy, test_env, test_policy, seeds, record_screenshots):
        env = self._env

        # initialize episodes
        if seeds is None:
            seeds = EpisodeGenerator._get_random_seeds(env.num_instances)
        states = env.reset(seeds=seeds, mode=('test' if test_env else 'train'),
                record_screenshots=record_screenshots)
        assert not env.died, 'Environment died'

        episodes = [Episode() for _ in xrange(len(states))]

        for step in xrange(self._max_steps_per_episode + 1):
            # Give up if you've reached episode limit
            if step == self._max_steps_per_episode:
                actions = [MiniWoBTerminate() if state is not None
                           else None for state in states]
            else:
                actions = policy.act(states, test_policy)

            # send attention weights to env's visualizer
            if policy.has_attention and step % self._visualize_attention_interval == 0:
                env.visualize_attention(policy.action_attention)

            next_states, rewards, dones, info = env.step(actions)
            assert not env.died, 'Environment died'

            # Update episodes
            assert len(episodes) == len(states) == len(actions) == len(rewards) == len(info['n'])
            for episode, (state, action, reward, metadata) in zip(
                    episodes, zip(states, actions, rewards, info['n'])):
                if action is not None:
                    episode.append(Experience(state, action, reward, metadata))

            states = next_states
            # TODO(kelvin): Pass old hidden state forward

            if all(dones):
                break

        policy.end_episodes()

        return episodes


class EpisodeWithJustification(Episode):
    def __init__(self, iterable, justification):
        super(EpisodeWithJustification, self).__init__(iterable)
        self._justification = justification

    @property
    def justification(self):
        """Return a Justification object."""
        return self._justification
