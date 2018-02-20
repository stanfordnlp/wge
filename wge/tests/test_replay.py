import random
from collections import Counter
import numpy as np
import pytest

from wge.replay import GroupedReplayBuffer, RewardPrioritizedReplayBuffer, UniformReplayBuffer
from wge.rl import Episode


class DummyEpisode(Episode):
    """Manually specify the reward for this Episode.

    Used to test reward-prioritized replay buffer.
    """

    def __init__(self, r):
        super(DummyEpisode, self).__init__()
        self._reward = r

    def discounted_return(self, t, gamma):
        return self._reward


class TestRewardPrioritizedReplayBuffer(object):
    def test_eviction(self):
        buff = RewardPrioritizedReplayBuffer(max_size=5, sampling_quantile=0.25, discount_factor=1.0)
        eps = [DummyEpisode(r) for r in range(-4, 4)]  # [-4, -3, ..., 2, 3], 8 total
        buff.extend(eps)

        # only highest reward eps should remain
        assert set([3, 2, 1, 0, -1]) == set(ep.discounted_return(0, 1.) for ep in buff._episodes)

        # if we add the same episodes again, we should see only 3s, 2s and 1s
        buff.extend(eps)
        assert set([3, 3, 2, 2, 1]) == set(ep.discounted_return(0, 1.) for ep in buff._episodes)
        assert len(buff) == 5

    def test_prioritized_sampling(self):
        np.random.seed(0)
        random.seed(0)

        buff = RewardPrioritizedReplayBuffer(max_size=4, sampling_quantile=0.25, discount_factor=1.0)
        buff.extend([DummyEpisode(r) for r in range(4)])  # [0, 1, 2, 3]
        eps, _, _ = buff.sample(500)
        sampled_rewards = [ep.discounted_return(0, 1.) for ep in eps]
        assert set(sampled_rewards) == {3}

        buff = RewardPrioritizedReplayBuffer(max_size=4, sampling_quantile=0.5, discount_factor=1.0)
        buff.extend([DummyEpisode(r) for r in range(4)])  # [0, 1, 2, 3]
        eps, _, _ = buff.sample(500)
        sampled_rewards = [ep.discounted_return(0, 1.) for ep in eps]
        assert set(sampled_rewards) == {2, 3}

        proportion_2 = len([r for r in sampled_rewards if r == 2]) / float(len(sampled_rewards))
        assert abs(proportion_2 - 0.5) <= 0.05


class TestGroupedReplayBuffer(object):
    def test_sample_from_groups(self):
        episodes = [1, 2, 24, 20, 11, 26, 2, 2]
        episode_grouper = lambda i: i / 10
        episode_identifier = lambda x: x

        buffer = GroupedReplayBuffer(episode_grouper, episode_identifier, UniformReplayBuffer)

        n = 1000000
        buffer.extend(episodes)
        samples, reported_probs, trace = buffer.sample(n)

        # compute empirical probs
        counts = Counter(samples)
        total = float(sum(counts.values()))
        probs = {}
        for i in counts:
            probs[i] = counts[i] / total

        # should be 3 groups
        # [1, 2, 2, 2]
        # [11]
        # [20, 24, 26]

        true_probs = {
            1: 1. / 12,
            2: 3. / 12,
            11: 1. / 3,
            20: 1. / 9,
            24: 1. / 9,
            26: 1. / 9,
        }

        # check that empirical probs match true probs
        def assert_close(episode):
            assert np.isclose(true_probs[episode], probs[episode], rtol=0.,
                              atol=0.001)  # should not be off by more than 0.1%

        assert len(probs) == 6
        for ep in [1, 2, 11, 20, 24, 26]:
            assert_close(ep)
