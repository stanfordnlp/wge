"""Define all the core classes for an RL problem."""
from abc import ABCMeta, abstractmethod, abstractproperty
from operator import itemgetter

import logging
import numpy as np

from collections import namedtuple, MutableSequence

from torch.autograd import Variable
from torch.nn import Module

from gtd.utils import cached_property, softmax


class State(object):
    pass


class HiddenState(object):
    pass


class Action(object):
    __metaclass__ = ABCMeta

    @property
    def justification(self):
        """Return a Justification object."""
        try:
            return self._justification
        except AttributeError:
            return None

    @justification.setter
    def justification(self, j):
        self._justification = j

    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError()

    @abstractmethod
    def __hash__(self):
        raise NotImplementedError()

    def __ne__(self, other):
        return not self.__eq__(other)


class Trace(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def to_json_dict(self):
        raise NotImplementedError

    @abstractmethod
    def dumps(self):
        """Return a unicode representation of the object."""
        raise NotImplementedError()


class Justification(Trace):
    """An object containing debug information explaining why an Action was chosen by the Policy."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def dumps(self):
        """Return a unicode representation of the object."""
        raise NotImplementedError()


class BaseExperience(object):
    pass  # marker class


class Experience(namedtuple('Experience', ['state', 'action', 'undiscounted_reward', 'metadata']),
                 BaseExperience):
    pass


class ScoredExperience(namedtuple('ScoredExperience',
    ['state', 'action', 'undiscounted_reward', 'log_prob', 'state_value', 'metadata']),
    BaseExperience):
    """Like an Experience, but with an extra log_prob attribute.

    Attributes:
        state (State)
        action (Action)
        undiscounted_reward (float)
        log_prob (Variable): a scalar Variable
        state_value (Variable): a scalar Variable
        metadata (dict[str->any])
    """
    pass


class Episode(MutableSequence):
    """A list of Experiences."""
    __slots__ = ['_experiences']
    discount_negative_reward = False

    def __init__(self, iterable=None):
        self._experiences = []
        if iterable:
            for item in iterable:
                self.append(item)

    @classmethod
    def configure(cls, discount_negative_reward):
        cls.discount_negative_reward = discount_negative_reward

    def __getitem__(self, item):
        return self._experiences[item]

    def __setitem__(self, key, value):
        assert isinstance(value, BaseExperience)
        self._experiences[key] = value

    def __delitem__(self, key):
        del self._experiences[key]

    def __len__(self):
        return len(self._experiences)

    def insert(self, index, value):
        assert isinstance(value, BaseExperience)
        return self._experiences.insert(index, value)

    def append(self, experience):
        assert isinstance(experience, BaseExperience)
        self._experiences.append(experience)

    def discounted_return(self, t, gamma):
        """Returns G_t, the discounted return.

        Args:
            t (int): index of the episode (supports negative indexing from
                back)
            gamma (float): the discount factor

        Returns:
            float
        """
        def discounted_reward(undiscounted, index):
            if undiscounted < 0 and not Episode.discount_negative_reward:
                return undiscounted
            else:
                return undiscounted * np.power(gamma, index)

        if t < -len(self._experiences) or t > len(self._experiences):
            raise ValueError("Index t = {} is out of bounds".format(t))

        return sum(discounted_reward(experience.undiscounted_reward, i)
                   for i, experience in enumerate(self._experiences[t:]))

    def __str__(self):
        experiences = "{}".format(self._experiences)[:50]
        return "Episode({}..., undiscounted return: {})".format(
                experiences, self.discounted_return(0, 1.0))
    __repr__ = __str__


class ScoredEpisode(Episode):
    """An Episode holding ScoredExperiences, rather than plain Experiences.
    
    Mostly just a marker class.
    """
    pass


class ActionScores(object):
    """A map from Actions to scores (log probabilities)."""
    def __init__(self, d, state_value):
        """Construct ActionScores.
        
        Args:
            d (dict[Action, Variable[FloatTensor]]): a map from Actions to scalar PyTorch Variables.
            state_value (Variable[FloatTensor]): a scalar value for the state we are at
        """
        for v in d.values():
            assert isinstance(v, Variable)
        assert isinstance(state_value, Variable)

        self._vars = d
        self._floats = {action: v.data.cpu()[0] for action, v in self._vars.items()}
        self._state_value = state_value

    @property
    def state_value(self):
        return self._state_value

    @property
    def as_floats(self):
        return self._floats

    @property
    def as_variables(self):
        return self._vars

    @property
    def best_action(self):
        if len(self.as_floats) == 0:
            return None
        return max(self.as_floats.items(), key=itemgetter(1))[0]

    def sample_action(self):
        actions, log_probs = zip(*self.as_floats.items())
        probs = softmax(log_probs)
        # TODO: Bring this back when the action score hack is fixed
        #if not np.isclose(total, 1.0):
        #    logging.warn('Action probs do not sum to 1: {} (will normalize).'.format(total))

        return np.random.choice(actions, p=probs)

    @property
    def justification(self):
        """Return a Justification object."""
        try:
            return self._justification
        except AttributeError:
            return None

    @justification.setter
    def justification(self, j):
        self._justification = j


class Policy(Module):
    """A parameterized RL policy mapping states to actions."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def act(self, states, test=False):
        """Given a batch of states, return a corresponding batch of actions.

        Args:
            states (list[State])
            test (bool): True if test time (False for train time)

        Returns:
            actions (list[Action])
        """
        raise NotImplementedError()

    @abstractmethod
    def score_actions(self, states, test=False):
        """Given a batch of states, return a corresponding batch of action scores (log probs).
        
        Args:
            states (list[State])
            test (bool): True if test time (False for train time)

        Returns:
            action_scores_batch (list[ActionScores]): a batch of ActionScores
        """
        raise NotImplementedError()

    @abstractmethod
    def update_from_episodes(self, episodes, gamma, take_grad_step):
        """Updates the Policy based on a batch of episodes.

        Args:
            episodes (list[Episode])
            gamma (float): discount factor
            take_grad_step (Callable): takes a loss Variable and takes a
                gradient step on the loss
        """
        raise NotImplementedError()

    @abstractmethod
    def update_from_replay_buffer(self, replay, gamma):
        """Updates the Policy from sampling from the replay buffer.

        Args:
            replay (ReplayBuffer):
            gamma (float): discount factor
            take_grad_step (Callable): takes a loss Variable and takes a
                gradient step on the loss
        
        Returns:
            trace (Trace)
        """
        raise NotImplementedError()

    @abstractmethod
    def end_episodes(self):
        """Signals the end of episodes to reset any episode state."""
        raise NotImplementedError()

    @abstractproperty
    def has_attention(self):
        """Returns True if the property action_attention is implemented."""
        raise NotImplementedError()

    @property
    def action_attention(self):
        raise NotImplementedError()
