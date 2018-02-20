import json
from os.path import join
from codecs import open
import numpy as np

from pyhocon import ConfigFactory

from gtd.io import makedirs
from wge.miniwob.trace import MiniWoBEpisodeTrace


AVG_REWARD = 'avg_reward'
AVG_EP_LENGTH = 'avg_episode_length'
SUCCESS_RATE = 'success_rate'
REPLAY_BUFFER_SIZE = 'replay_buffer_size'
REPLAY_LOSS = 'replay_loss'
REPLAY_LOSS_AS_PROB = 'replay_loss_as_prob'


class Stat(object):
    def __init__(self, stat_type, latest=None, last_seen=None,
                 best=None, hit_times=None):
        """Construct statistic.
        
        Args:
            stat_type (str): type of stat
        """
        if stat_type == SUCCESS_RATE:
            ascending = True
            # [0.1, 0.2, ..., 1.0]
            targets = [round(0.1 * i, 1) for i in range(1, 11, 1)]
        elif stat_type == AVG_REWARD:
            ascending = True
            # [-1.0, -0.9, ..., 0.9, 1.0]
            targets = [round(0.1 * i, 1) for i in range(-10, 11, 1)]
        elif stat_type == AVG_EP_LENGTH:
            ascending = False
            targets = range(10)
        elif stat_type == REPLAY_BUFFER_SIZE:
            ascending = True
            # [100, 200, ..., 1000]
            targets = [100 * i for i in range(1, 11)]
        elif stat_type == REPLAY_LOSS:
            ascending = False
            ps = np.arange(0.05, 1, 0.1)  # [.05, .15, ..., 0.95]
            targets = [round(-np.log(p), 2) for p in ps]
        elif stat_type == REPLAY_LOSS_AS_PROB:
            ascending = True
            targets = [round(p, 2) for p in np.arange(0.05, 1, 0.1)]  # [.05, .15, ..., 0.95]
        else:
            raise ValueError(stat_type)
        self._targets = targets
        self._ascending = ascending

        # latest and best either both exist, or both don't exist
        if latest is None:
            assert best is None
        else:
            assert best is not None

        self.stat_type = stat_type
        self.latest = latest
        self.last_seen = last_seen
        self.best = best

        if hit_times is None:
            hit_times = {}
        self.hit_times = hit_times

    def add_value(self, value, train_step):
        self.latest = value
        self.last_seen = train_step

        if self._ascending:
            hit = lambda v, target: v >= target
            best = lambda a, b: max(a, b)
        else:
            hit = lambda v, target: v <= target
            best = lambda a, b: min(a, b)

        # compute new best
        if self.best is None:
            self.best = value
        else:
            self.best = best(self.best, value)

        # update hitting times
        for target in self._targets:
            if target in self.hit_times:
                continue  # already hit

            if hit(value, target):
                self.hit_times[target] = train_step

    def to_json_dict(self):
        attrs = ['stat_type', 'latest', 'last_seen', 'best', 'hit_times']
        return {k: getattr(self, k) for k in attrs}

    @classmethod
    def from_json_dict(cls, d):
        return Stat(**d)

    def to_config_tree(self):
        d = self.to_json_dict()
        # need to convert hit_times to (key, value) pairs,
        # since a float can't be a key in HOCON format
        items = d['hit_times'].items()
        items = sorted(items, reverse=True)
        items = [list(item) for item in items]  # json doesn't like tuples
        d['hit_times'] = items
        return ConfigFactory.from_dict(d)

    @classmethod
    def from_config_tree(cls, tree):
        tree['hit_times'] = dict(tree['hit_times'])
        return cls.from_json_dict(tree)

    @classmethod
    def update_metadata(cls, meta, stat_path, stat_type, new_value, train_step):
        """Update the value of a Stat stored in a Metadata object.
        
        Args:
            meta (Metadata)
            stat_path (str)
            stat_type (str)
            new_value (float)
            train_step (int)
        """
        try:
            # load
            stat_config_tree = meta[stat_path]._config_tree
            stat = Stat.from_config_tree(stat_config_tree)
        except KeyError:
            # initialize
            stat = Stat(stat_type)

        stat.add_value(new_value, train_step)  # update
        meta[stat_path] = stat.to_config_tree()  # save


class EpisodeLogger(object):
    def __init__(self, trace_dir, tb_logger, metadata):
        """
        
        Args:
            trace_dir (str)
            tb_logger (tensorboard_logger.Logger)
            metadata (Metadata)
        """
        self.trace_dir = trace_dir
        self.tb_logger = tb_logger
        self.metadata = metadata

    def __call__(self, episodes, label, train_step, log_traces):
        if log_traces:
            self._log_traces(episodes, label, train_step)
        self._log_stats(episodes, label, train_step)

    def _log_traces(self, episodes, label, train_step):
        trace_dir = join(self.trace_dir, label)
        makedirs(trace_dir)
        trace_path = join(trace_dir, str(train_step))

        episode_traces = [MiniWoBEpisodeTrace(ep) for ep in episodes]

        # save machine-readable version
        with open(trace_path + '.json', 'w', 'utf8') as f:
            trace_dicts = [trace.to_json_dict() for trace in episode_traces]
            json.dump(trace_dicts, f, indent=2)

        # save pretty-printed version
        with open(trace_path + '.txt', 'w', 'utf8') as f:
            for i, trace in enumerate(episode_traces):
                f.write("=" * 25 + " EPISODE {} ".format(i) + "=" * 25)
                f.write('\n\n')
                f.write(trace.dumps())
                f.write('\n\n')

        # save screenshots
        for i, ep in enumerate(episodes):
            if not self._has_screenshot(ep):
                continue
            img_path = trace_path + '-img'
            makedirs(img_path)
            actions = []
            for j, experience in enumerate(ep):
                state, action = experience.state, experience.action
                path = join(img_path, '{}-{}-{}.png'.format(train_step, i, j))
                state.screenshot.save(path)
                actions.append(action.to_dict())
            # write action summary
            path = join(img_path, '{}-{}.json'.format(train_step, i))
            with open(path, 'w') as fout:
                json.dump(actions, fout)

    def _has_screenshot(self, episode):
        """Return whether the episode contains screenshot data."""
        if not episode:
            return False
        state = episode[0].state
        return hasattr(state, 'screenshot') and state.screenshot is not None

    def _log_stats(self, episodes, label, train_step):
        raw_stats = self._compute_raw_stats(episodes)

        meta = self.metadata
        for stat_type, value in raw_stats.items():
            # update tboard
            self.tb_logger.log_value('{}_{}'.format(label, stat_type),
                                     value=value, step=train_step)
            # update metadata
            stat_path = '.'.join(['stats', label, stat_type])
            Stat.update_metadata(meta, stat_path, stat_type, value, train_step)

    def _compute_raw_stats(self, episodes):
        episode_lengths = [float(len(ep)) for ep in episodes]
        returns = [ep.discounted_return(0, gamma=1.0) for ep in episodes]

        # as defined in the WoB paper
        num_positive = len([r for r in returns if r > 0])
        num_non_zero = len([r for r in returns if r != 0])
        success_rate = float(num_positive) / max(num_non_zero, 1.)

        avg = lambda seq: sum(seq) * 1. / max(len(seq), 1)

        return {
            SUCCESS_RATE: success_rate,
            AVG_REWARD: avg(returns),
            AVG_EP_LENGTH: avg(episode_lengths)
        }


class ReplayLogger(object):
    def __init__(self, trace_dir, tb_logger, metadata):
        """

        Args:
            trace_dir (str)
            tb_logger (tensorboard_logger.Logger)
            metadata (Metadata)
        """
        self.trace_dir = trace_dir
        self.tb_logger = tb_logger
        self.metadata = metadata

    def __call__(self, buffer_size, buffer_status, replay_loss, replay_trace,
                 control_step, log_trace):
        """
        
        Args:
            buffer_size (int)
            buffer_status (str)
            replay_loss (float)
            replay_trace (ReplayBufferUpdateTrace)
            control_step (int)
            log_trace (bool)
        """
        # log buffer size
        self._log_stat(REPLAY_BUFFER_SIZE, buffer_size, control_step)

        # log buffer status
        self.metadata['stats.replay_buffer_status'] = buffer_status

        # log replay loss
        if replay_loss is not None:
            self._log_stat(REPLAY_LOSS, replay_loss, control_step)
            replay_loss_as_prob = np.exp(-replay_loss)
            self._log_stat(REPLAY_LOSS_AS_PROB, replay_loss_as_prob, control_step)

        # write trace to file
        if log_trace:
            trace_path = join(self.trace_dir, str(control_step)) + '.txt'
            with open(trace_path, 'w', 'utf8') as f:
                if replay_trace is None:
                    f.write('None')
                else:
                    f.write(replay_trace.dumps())

    def _log_stat(self, stat_type, value, step):
        self.tb_logger.log_value(stat_type, value, step)
        stat_path = 'stats.' + stat_type
        Stat.update_metadata(self.metadata, stat_path, stat_type, value, step)
