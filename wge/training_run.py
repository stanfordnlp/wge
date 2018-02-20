import os
import random
from os.path import dirname, realpath, join
from codecs import open

import logging
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from gtd.ml.torch.training_run import TorchTrainingRun
from gtd.ml.torch.utils import try_gpu, random_seed
from gtd.ml.training_run import TrainingRuns
from gtd.utils import as_batches
from wge import data
from wge.environment import Environment
from wge.episode_generator import BasicEpisodeGenerator
from wge.log import EpisodeLogger, ReplayLogger
from wge.miniwob.action import MiniWoBTerminate
from wge.miniwob.demonstrations import load_demonstrations
from wge.miniwob.perfect_oracle import get_perfect_oracle
from wge.miniwob.program_oracle import get_program_oracle
from wge.miniwob.program_policy import ProgramPolicy
from wge.miniwob.labeled_demonstration import LabeledDemonstration
from wge.miniwob.reward import get_reward_processor
from wge.replay import RewardPrioritizedReplayBuffer, \
    GroupedReplayBuffer
from wge.rl import Episode
from wge.wob_policy import MiniWoBPolicy


class MiniWoBTrainingRuns(TrainingRuns):
    def __init__(self, check_commit=True):
        data_dir = data.workspace.experiments           # pylint: disable=no-member
        src_dir = dirname(dirname(realpath(__file__)))  # root of the Git repo
        super(MiniWoBTrainingRuns, self).__init__(
            data_dir, src_dir, MiniWoBTrainingRun, check_commit=check_commit)


class MiniWoBTrainingRun(TorchTrainingRun):
    """Encapsulates the elements of a training run."""

    def __init__(self, config, save_dir):
        super(MiniWoBTrainingRun, self).__init__(config, save_dir)
        self.workspace.add_dir('traces_replay', join('traces', 'replay'))
        self.workspace.add_file('traces_demo', join('traces', 'demo-parse-log.txt'))

        # need to make sure that these times coincide
        assert config.log.trace_evaluate % config.log.evaluate == 0
        assert (config.log.explore % config.explore.program == 0 or
                config.log.explore % config.explore.neural == 0)
        assert config.log.replay % config.train.replay == 0
        assert config.log.trace_replay % config.log.replay == 0
        assert config.log.trace_explore % config.log.explore == 0

        # construct environment
        Episode.configure(config.discount_negative_reward)
        env = Environment.make(config.env.domain, config.env.subdomain)  # TODO: Refactor into a get_environment
        env.configure(
            num_instances=config.env.num_instances, seeds=range(config.env.num_instances), headless=config.env.headless,
            base_url=os.environ.get("MINIWOB_BASE_URL"),
            cache_state=False,  # never cache state
            reward_processor=get_reward_processor(config.env.reward_processor),
            wait_ms=config.env.wait_ms,
            block_on_reset=config.env.block_on_reset,
            refresh_freq=config.env.refresh_freq,
        )
        self._env = env

        # construct episode generators
        self._basic_episode_generator = BasicEpisodeGenerator(self._env,
                                        config.explore.max_steps_per_episode,
                                        config.log.visualize_attention)

        def state_equality_checker(s1, s2):
            """Compare two State objects."""
            r1 = s1.dom.visualize() if s1 else None
            r2 = s2.dom.visualize() if s2 else None
            return r1 == r2
            # TODO(kelvin): better equality check

        # construct episode logger
        trace_dir = join(self.workspace.root, 'traces')
        self._episode_logger = EpisodeLogger(trace_dir, self.tb_logger,
                                             self.metadata)

        # construct replay buffer

        # group episodes by query fields
        episode_grouper = lambda ep: frozenset(ep[0].state.fields.keys)
        episode_identifier = lambda ep: id(ep)

        # each has its own buffer
        group_buffer_factory = lambda: RewardPrioritizedReplayBuffer(
            max_size=config.replay_buffer.size,
            sampling_quantile=1.0,
            discount_factor=config.gamma)

        # buffers are combined into a single grouped buffer
        self._replay_buffer = GroupedReplayBuffer(
            episode_grouper, episode_identifier,
            group_buffer_factory, min_group_size=config.replay_buffer.min_size)

        self._replay_steps = config.train.replay_steps
        self._gamma = config.gamma

        # construct replay logger
        self._replay_logger = ReplayLogger(self.workspace.traces_replay,
                self.tb_logger, self.metadata)

        # load demonstrations
        with open(self.workspace.traces_demo, 'w', 'utf8') as fout:     # pylint: disable=no-member
            # NOTE: this may be an empty list for some tasks
            self._demonstrations = load_demonstrations(
                    config.env.subdomain, config.demonstrations.base_dir,
                    config.demonstrations.parser, logfile=fout,
                    min_raw_reward=config.demonstrations.min_raw_reward)

            # keep a random subset of demonstrations
            with random_seed(0):
                random.shuffle(self._demonstrations)
            self._demonstrations = self._demonstrations[:config.demonstrations.max_to_use]

        num_demonstrations = len(self._demonstrations)
        self.metadata['stats.num_demonstrations'] = num_demonstrations
        if num_demonstrations == 0:
            logging.warn('NO DEMONSTRATIONS AVAILABLE')

        # build neural policy
        neural_policy = try_gpu(MiniWoBPolicy.from_config(config.policy))
        optimizer = optim.Adam(neural_policy.parameters(),
                               lr=config.train.learning_rate)

        # TODO: reload replay buffer?
        self.train_state = self.checkpoints.load_latest(
                neural_policy, optimizer)

        # build program policy
        self._program_policy = self._build_program_policy()

    def close(self):
        self._env.close()

    def _trace_path(self, dir_path):
        """Construct a path to a trace file, based on # of train steps."""
        return join(dir_path, str(self.train_state.train_steps))

    @property
    def train_state(self):
        return self._train_state

    @train_state.setter
    def train_state(self, ts):
        self._train_state = ts

    @property
    def neural_policy(self):
        return self.train_state.model

    @property
    def program_policy(self):
        return self._program_policy

    def train(self):
        config = self.config

        # TODO(kelvin): is this accessing the right train_state?
        take_grad_step = lambda loss: self._take_grad_step(self.train_state, loss)

        # track best reward, for early stopping
        best_avg_reward_train = float('-inf')
        best_bc_ckpt = None  # checkpoint number (grad steps) of the checkpoint with best reward for BC
        pretraining_stage = self.config.train.behavioral_cloning
        # indicates whether we are still in pre-training stage
        # switches to False once pretraining starts to overfit
        # (best_bc_reward begins to drop)

        for control_step in tqdm(xrange(config.train.max_control_steps), desc='Outer training loop'):

            # plot number of grad steps taken against control steps
            # - control_step is used to determine the timing of various events
            # - train_state.train_steps is the number of grad steps we've taken
            if control_step % 10 == 0:
                self.tb_logger.log_value('grad_steps',
                    self.train_state.train_steps, step=control_step)

            self.metadata['control_steps'] = control_step

            if pretraining_stage:
                # Behavioral cloning
                self._behavioral_cloning(self.neural_policy, self._demonstrations)
            else:

                # explore and update program policy
                if (control_step % config.explore.program == 0) and \
                    self.program_policy is not None:
                    episodes = self._explore(self.program_policy,
                                             'explore_program',
                                             control_step)

                    if config.train.reinforce_program:
                        self.program_policy.update_from_episodes(episodes,
                                                    self._gamma, take_grad_step)

                # explore and update neural policy
                if control_step % config.explore.neural == 0:
                    episodes = self._explore(self.neural_policy,
                                             'explore_neural',
                                             control_step)

                    if config.train.reinforce_neural:
                        self.neural_policy.update_from_episodes(episodes,
                                                    self._gamma, take_grad_step)

                # update neural policy from replay buffer
                if control_step % config.train.replay == 0:
                    log_replay = (control_step % config.log.replay == 0)
                    trace_replay = (control_step % config.log.trace_replay == 0)
                    self._replay_episodes(self.neural_policy, control_step,
                                          log_replay, trace_replay)

            # evaluate program and neural policy
            if control_step % config.log.evaluate == 0:
                trace_evaluate = (control_step % config.log.trace_evaluate == 0)

                self._evaluate(self.neural_policy,
                    config.log.episodes_to_evaluate_small, 'test',
                    test_env=True,
                    log=True, trace=trace_evaluate, control_steps=control_step)

                if self.program_policy is not None:
                    self._evaluate(self.program_policy,
                        config.log.episodes_to_evaluate_small, 'test_program',
                        test_env=True,
                        log=True, trace=trace_evaluate, control_steps=control_step)

            # bigger evaluation of neural policy
            if (control_step % config.log.evaluate_big == 0) and (control_step != 0):
                avg_reward_test = self._evaluate(self.neural_policy,
                                config.log.episodes_to_evaluate_big, 'test_big',
                                test_env=True,
                                log=True, trace=False, control_steps=control_step)
                # (don't trace, because it is too large)

                avg_reward_train = self._evaluate(self.neural_policy,
                                config.log.episodes_to_evaluate_big, 'train_big',
                                test_env=False,  # eval on TRAIN environment
                                log=True, trace=False, control_steps=control_step)

                if pretraining_stage:
                    if avg_reward_train > best_avg_reward_train:
                        print 'PRE-TRAINING -- new high at step {}: {:.2f}'.format(
                            control_step, avg_reward_train)

                        self.metadata['stats.best_avg_reward_train_bc.value'] = avg_reward_train
                        self.metadata['stats.best_avg_reward_train_bc.hit_time'] = control_step

                        # save model with best BC reward
                        if best_bc_ckpt is not None:
                            # delete old best checkpoint
                            self.checkpoints.delete(best_bc_ckpt)
                        # save new best checkpoint
                        self.checkpoints.save(self.train_state)
                        best_bc_ckpt = self.train_state.train_steps

                    elif avg_reward_train <= best_avg_reward_train:
                        print 'PRE-TRAINING -- overfit at step {}: {:.2f} (best={:.2f})'.format(
                            control_step, avg_reward_train, best_avg_reward_train)

                        # if latest reward is worse than best, stop pretraining
                        pretraining_stage = False

                        # roll back to best behavior cloning checkpoint
                        # print 'PRE-TRAINING -- rolling back to best at {}'.format(best_bc_ckpt)
                        # self.train_state = self.checkpoints.load(best_bc_ckpt,
                        #                                          self.train_state.model,
                        #                                          self.train_state.optimizer)
                        #
                        # # completely reset the optimizer
                        # new_optimizer = optim.Adam(self.train_state.model.parameters(),
                        #                            lr=config.train.learning_rate)
                        # self.train_state.optimizer = new_optimizer

                if avg_reward_train > best_avg_reward_train:
                    best_avg_reward_train = avg_reward_train
                    # note that we are saving avg_reward_test here!
                    self.metadata['stats.best_avg_reward.value'] = avg_reward_test
                    self.metadata['stats.best_avg_reward.hit_time'] = control_step

                # stop training if we reach max reward
                if np.isclose(avg_reward_train, 1.0):
                    break

            # save neural policy
            if control_step % config.log.save == 0 and control_step != 0:
                print 'Saving checkpoint'
                self.checkpoints.save(self.train_state)


    def _filter_episodes(self, episodes):
        return [ep for ep in episodes if not isinstance(ep[-1].action, MiniWoBTerminate)]

    def _explore(self, policy, label, control_step):
        # roll out episodes using basic generator and program policy
        episodes = self._basic_episode_generator(policy, test_env=False,
                                                 test_policy=False)

        # Update replay buffer (excluding episodes that ended with MiniWoBTerminate)
        self._replay_buffer.extend(self._filter_episodes(episodes))

        # log episodes
        if control_step % self.config.log.explore == 0:
            trace_explore = control_step % self.config.log.trace_explore == 0
            self._episode_logger(episodes, label,
                                 control_step, trace_explore)
        return episodes

    def _evaluate(self, policy, num_episodes, label, test_env,
                  log, trace, control_steps):
        """Evaluates policy with test time flag set to True on some episodes.
        Returns list of rewards and episode lengths

        Args:
            policy (Policy)
            num_episodes (int)
            label (str)
            test_env (bool)
            log (bool)
            trace (bool)
            control_steps (int)
        
        Returns:
            avg_return (float)
        """
        num_instances = self._env.num_instances
        eval_iters = num_episodes / num_instances
        all_seeds = range(1, eval_iters * num_instances + 1)  # [1, 2, 3, ...]
        episodes = []
        for _ in tqdm(xrange(eval_iters), desc="Evaluating policy"):
            seeds = [all_seeds.pop() for _ in range(num_instances)]

            # use different seeds for testing
            if test_env:
                seeds = [seed * -1. for seed in seeds]

            episodes.extend(self._basic_episode_generator(
                policy, test_env=test_env, test_policy=True,
                seeds=seeds, record_screenshots=self.config.log.record_screenshots))

        if log:
            self._episode_logger(episodes, label, control_steps,
                                 log_traces=trace)

        avg = lambda seq: sum(seq) / max(len(seq), 1)
        returns = [ep.discounted_return(0, gamma=1.0) for ep in episodes]
        return avg(returns)

    def _replay_episodes(self, policy, control_step, log, trace):
        """Updates the policy by sampling from the replay buffer.

        Args:
            policy (Policy): the policy to update
            control_step (int): the current control step
            log (bool): whether to log basic stats
            trace (bool): whether to log trace
        """
        buff_size = len(self._replay_buffer)
        buff_status = self._replay_buffer.status()

        # need to initialize, because for-loop below may be empty
        replay_trace = None
        replay_loss = None
        take_grad_step = lambda loss: self._take_grad_step(self.train_state, loss)

        progress_msg = "Replaying Episodes (buffer size {})".format(buff_size)
        for _ in tqdm(xrange(self._replay_steps), desc=progress_msg):
            replay_loss, replay_trace = policy.update_from_replay_buffer(
                self._replay_buffer, self._gamma, take_grad_step)

        if log:
            self._replay_logger(buff_size, buff_status, replay_loss,
                                replay_trace, control_step, trace)

    def _behavioral_cloning(self, policy, demonstrations):
        """Perform behavioral cloning on the demonstrations.

        Args:
            policy (Policy)
            demonstrations (list[EpisodeGraph])
        """
        # TODO: Don't hard-code batch size
        if len(demonstrations) > 0:
            batch = np.random.choice(demonstrations, size=5, replace=True)
            policy.update_from_demonstrations(
                    batch, lambda loss: self._take_grad_step(
                        self.train_state, loss))

    def _build_program_policy(self):
        if self.config.program_policy.type == "program":
            demos = self._demonstrations
            if len(demos) == 0:
                return None
            else:
                labeled_demos = [
                    LabeledDemonstration.from_episode_graph(episode_graph)
                    for episode_graph in tqdm(demos, desc="Inducing programs")]
            return ProgramPolicy.from_config(labeled_demos, self.config.program_policy)
        elif self.config.program_policy.type == "program-oracle":
            return get_program_oracle(
                    self.config.env.subdomain, self.config.program_policy)
        elif self.config.program_policy.type == "perfect-oracle":
            return get_perfect_oracle(
                    self.config.env.subdomain, self.config.program_policy)
        else:
            raise ValueError("{} not a supported program policy".format(
                self.config.program_policy.type))
