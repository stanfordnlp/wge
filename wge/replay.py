import random
from abc import ABCMeta, abstractmethod
from collections import defaultdict, Counter, OrderedDict

import math
import numpy as np

from gtd.log import indent
from wge.rl import Trace


def normalize_counts(counts):
    """Return a normalized Counter object."""
    normed = Counter()
    total = float(sum(counts.values(), 0.0))
    assert total > 0  # cannot normalize empty Counter
    for key, ct in counts.iteritems():
        normed[key] = ct / total
    return normed


class ReplayBuffer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def sample(self, num_episodes):
        """Sample WITH replacement from the buffer.

        Args:
            num_episodes (int): number of episodes to return.

        Returns:
            sampled_episodes (list[Episode])
            sample_probs (list[float]): probability of sampling the episode
            trace (ReplayBufferSampleTrace)
        """
        raise NotImplementedError

    @abstractmethod
    def extend(self, episodes):
        """Extends the buffer with the given episodes.

        Randomly evicts episodes from the buffer as necessary.

        Args:
            episodes (list[Episode])
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def status(self):
        """A human-readable string describing the status of the buffer."""
        raise NotImplementedError


class UniformReplayBuffer(ReplayBuffer):
    """Minimalist replay buffer."""

    def __init__(self):
        self._episodes = []

    def __len__(self):
        return len(self._episodes)

    def sample(self, num_episodes):
        indices = np.random.choice(len(self._episodes), size=num_episodes, replace=True)
        episodes = [self._episodes[i] for i in indices]
        probs = [1.] * len(episodes)
        trace = None
        return episodes, probs, trace

    def extend(self, episodes):
        self._episodes.extend(episodes)

    def status(self):
        return 'size: {}'.format(len(self))


class ReplayBufferNotReadyException(Exception):
    pass


class RewardPrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, max_size, sampling_quantile, discount_factor):
        """RewardPrioritizedReplayBuffer.

        Lowest-reward episodes are evicted when the buffer becomes full.
        Buffer only samples from the top K-quantile of what it contains.

        (where K = sampling_quantile)

        Args:
            max_size (int): max size of the buffer. 
            sampling_quantile (float): should be in (0, 1]
            discount_factor (float)
        """
        self.max_size = max_size
        self.sampling_quantile = sampling_quantile
        self._discount_factor = discount_factor
        self._episodes = []  # this should always be sorted from highest-reward to lowest-reward

    def __len__(self):
        return len(self._episodes)

    def sample(self, num_episodes):
        n = len(self)
        if n == 0:
            raise RuntimeError('Cannot sample from an empty buffer.')

        # only sample as many as are contained in the buffer
        num_episodes = min(num_episodes, len(self))

        # only sample from the top k-quantile
        sample_limit = int(math.ceil(n * self.sampling_quantile))

        # if the top k-quantile isn't large enough to get num_episodes unique episodes, expand it
        sample_limit = max(sample_limit, num_episodes)

        # don't ever sample the same thing twice
        sample_indices = list(np.random.choice(sample_limit, size=num_episodes, replace=False))

        sample_episodes = [self._episodes[i] for i in sample_indices]
        sample_probs = [1.] * len(sample_episodes)
        # TODO(kelvin): similar to the old replay buffer, we are just hacking sample_probs to be all 1s right now
        trace = PrioritizedRewardReplayBufferTrace(self._episodes)
        return sample_episodes, sample_probs, trace

    def extend(self, episodes):
        # only add episodes with full reward
        episodes = [ep for ep in episodes if ep.discounted_return(0, 1.) == 1]
        # TODO(kelvin): just create a FullRewardOnlyBuffer, rather than
        # hacking RewardPrioritizedBuffer

        # DISABLED: only add episodes with positive reward
        # episodes = [ep for ep in episodes if ep.discounted_return(0, 1.) > 0]

        self._episodes.extend(episodes)

        if len(self._episodes) > self.max_size:
            # the sort in the following lines is an in-place sort
            # this shuffle breaks the in-place nature of that sort, which
            # would undesirably favor older episodes
            shuffled_episodes = list(self._episodes)
            random.shuffle(shuffled_episodes)

            sorted_episodes = sorted(shuffled_episodes, key=lambda ep: ep.discounted_return(0, 1.), reverse=True)

            self._episodes = sorted_episodes[:self.max_size]

    def status(self):
        if len(self) == 0:
            return 'empty'

        rewards = sorted(ep.discounted_return(0, 1.) for ep in self._episodes)
        median = rewards[len(rewards) / 2]
        min = rewards[0]
        max = rewards[-1]
        mean = sum(rewards) / len(rewards)

        return u'n={n:<4} mean={mean:.2f} range=[{min:.2f}, {max:.2f}] median={median:.2f}'.format(
            n=len(rewards), min=min, median=median, max=max, mean=mean)


class GroupedReplayBuffer(ReplayBuffer):
    """Buffer of Episodes to replay."""

    def __init__(self, episode_grouper, episode_identifier,
                 buffer_factory, min_group_size):
        """Construct replay buffer.

        WARNING:
            We assume that the probability of sampling an episode is just 1.

            Compared to using the real sample prob (which can be easily computed),
            this is more stable for downstream importance sampling.

            We already violate the assumptions of importance sampling, because our
            proposal distribution doesn't have full support over the target distribution.
            Exact sample probs actually exacerbate the problem.
            Approximate sample probs somewhat mitigate the problem.

        Args:
            episode_grouper (Callable[Episode, object]): see self._sample_from_groups
            episode_identifier (Callable[Episode, object]): see self._sample_from_groups
            buffer_factory (Callable[[], ReplayBuffer): creates a brand new buffer
            min_group_size (int): if a group's buffer is smaller than this size,
                we will not sample from it.
        """
        self._group_buffers = defaultdict(buffer_factory)
        self._episode_grouper = episode_grouper
        self._episode_identifier = episode_identifier
        self._min_group_size = min_group_size

    def sample(self, num_episodes):
        group_labels = [label for label, buffer in self._group_buffers.items()
                        if len(buffer) >= self._min_group_size]

        if len(group_labels) == 0:
            # none of the buffers are ready
            raise ReplayBufferNotReadyException()

        num_groups = len(group_labels)
        uniform_probs = [1. / num_groups] * num_groups
        group_counts = np.random.multinomial(num_episodes, uniform_probs)  # sample uniformly from groups

        sampled_episodes = []
        sample_probs = []
        traces = {}
        assert len(group_labels) == len(group_counts)
        for label, group_count in zip(group_labels, group_counts):
            group_buffer = self._group_buffers[label]
            eps, probs, trace = group_buffer.sample(group_count)
            sampled_episodes.extend(eps)
            sample_probs.extend(probs)
            traces[label] = trace

        group_counts_dict = dict(zip(group_labels, group_counts))
        full_trace = GroupedReplayBufferTrace(traces, group_counts_dict)

        return sampled_episodes, sample_probs, full_trace

    def extend(self, episodes):
        # group the episodes
        grouped_episodes = defaultdict(list)
        for ep in episodes:
            grouped_episodes[self._episode_grouper(ep)].append(ep)

        # add the episodes to their respective buffers
        for label, group in grouped_episodes.iteritems():
            self._group_buffers[label].extend(group)

    def __len__(self):
        return sum(len(buffer) for buffer in self._group_buffers.values())

    def status(self):
        if len(self._group_buffers) == 0:
            return 'empty'

        return '\n'.join(u'{}: {}'.format(buffer.status(), label)
                         for label, buffer in self._group_buffers.items())


class GroupedReplayBufferTrace(Trace):
    def __init__(self, group_traces, group_counts):
        def trace_sort_key(item):
            group_label, trace = item
            if isinstance(trace, PrioritizedRewardReplayBufferTrace):
                return -trace.mean  # sort by mean reward of group
            else:
                return repr(group_label)  # sort by group label

        self._group_traces = OrderedDict(sorted(group_traces.items(), key=trace_sort_key))
        self._group_counts = OrderedDict(sorted(group_counts.items(), key=lambda x: -x[1]))

    def to_json_dict(self):
        return {'group_counts': {repr(label): count for label, count in self._group_counts.items()},
                'group_traces': {repr(label): stat.to_json_dict() for label, stat in self._group_traces.items()}
                }

    def dumps(self):
        return u'group stats:\n{}\nsample counts:\n{}'.format(
            indent('\n'.join(u'{}: {}'.format(
                trace.dumps(), label) for label, trace in self._group_traces.items())),
            indent('\n'.join(u'{:<5}: {}'.format(c, k) for k, c in self._group_counts.items())),
        )


class PrioritizedRewardReplayBufferTrace(Trace):
    def __init__(self, episodes):
        self._rewards = sorted(ep.discounted_return(0, 1.) for ep in episodes)
        self.median = self._rewards[len(self._rewards) / 2]
        self.min = self._rewards[0]
        self.max = self._rewards[-1]
        self.mean = sum(self._rewards) / len(self._rewards)

    def dumps(self):
        return u'n={n:<4} mean={mean:.2f} range=[{min:.2f}, {max:.2f}] median={median:.2f}'.format(
            n=len(self._rewards), min=self.min, median=self.median, max=self.max, mean=self.mean)

    def to_json_dict(self):
        return {'median': self.median,
                'mean': self.mean,
                'min': self.min,
                'max': self.max
                }
