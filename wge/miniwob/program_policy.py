from abc import ABCMeta, abstractmethod
import logging

import numpy as np

from collections import OrderedDict, namedtuple
from gtd.log import indent
from wge.miniwob.action import MiniWoBTerminate
from wge.rl import Policy, Justification
from wge.miniwob.labeled_demonstration import WeightedProgram
from wge.miniwob.program import ExecutionEnvironment, \
        ProgramExecutionException, ElementSet, TerminateToken


class ProgramPolicy(Policy):
    __metaclass__ = ABCMeta

    def __init__(self, labeled_demonstrations, config):
        super(ProgramPolicy, self).__init__()
        self._config = config
        self._labeled_demos = labeled_demonstrations
        for labeled_demo in self._labeled_demos:
            self._init_weights(labeled_demo)
        self.end_episodes()

    @classmethod
    def from_config(cls, labeled_demonstrations, config):
        """Get a correct instance of ProgramPolicy.

        Args:
            labeled_demonstrations (list[LabeledDemonstration])
            config: program_policy section of the config
        """
        if config.parameterization == 'linear':
            return LinearProgramPolicy(labeled_demonstrations, config)
        elif config.parameterization == 'softmax':
            return SoftmaxProgramPolicy(labeled_demonstrations, config)
        else:
            raise ValueError('Unkwown parameterization {}'.format(
                config.parameterization))

    def score_actions(self, states, test=False):
        raise NotImplementedError()

    def act(self, states, test=False):
        if self._start_over:
            self._start_over = False
            demos = [self._select_demo(state) for state in states]
            self._demo_players = [DemoPlayer(demo, self, test) for demo in demos]

        actions = []
        assert len(states) == len(self._demo_players)
        for i, (state, demo_player) in enumerate(
                zip(states, self._demo_players)):
            if state is None:
                actions.append(None)
            else:
                action = demo_player.next_action(state)
                actions.append(action)

        return actions

    def end_episodes(self):
        self._start_over = True

    def update_from_episodes(self, episodes, gamma, take_grad_step):
        assert len(self._demo_players) == len(episodes)
        for demo_player, episode in zip(self._demo_players, episodes):
            self._update_weights(demo_player, episode)

    def update_from_replay_buffer(self, replay, gamma, take_grad_step):
        raise ValueError("Should never be updated from Replay Buffer.")

    @property
    def has_attention(self):
        return False

    def _select_demo(self, state):
        """Returns a DemoPlayer appropriate for this state. DemoPlayers are
        chosen based on the similarity of the fields to the state's fields.

        Args:
            state (MiniWoBState)

        Returns:
            DemoPlayer
        """
        def num_diff_unique_keys(first, second):
            return len(set(first.keys) ^ set(second.keys))

        def fields_similarity(first, second):
            return 1. / np.square(num_diff_unique_keys(first, second) + 1)

        # First check if there are any demos with the same keys
        fields = state.fields
        compatible_demos = [
            demo for demo in self._labeled_demos
            if num_diff_unique_keys(fields, demo.fields) == 0]
        if len(compatible_demos) != 0:
            return np.random.choice(compatible_demos)

        # If no exact match, weight the other demos
        similarities = [fields_similarity(
            fields, demo.fields) for demo in self._labeled_demos]
        weights = [s / sum(similarities) for s in similarities]

        return np.random.choice(self._labeled_demos, p=weights)

    @abstractmethod
    def _init_weights(self, labeled_demo):
        """Initialize the weights of each WeightedProgram.

        Args:
            labeled_demo (LabeledDemonstration)
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_program_probs(self, weighted_programs):
        """Return the probability distribution over the programs.

        Args:
            weighted_programs (list[WeightedProgram])

        Returns:
            list of length len(weighted_programs)
        """
        raise NotImplementedError()

    @abstractmethod
    def _update_weights(self, demo_player, episode):
        """Update the weights of the played program based on the episode.

        Args:
            demo_player (DemoPlayer)
            episode (Episode)
        """
        raise NotImplementedError()


class LinearProgramPolicy(ProgramPolicy):
    """Program policy with linear weighting scheme.

    - Initialize the weights uniformly
    - Sample the program proportionally to its weight
    - Update: bump up the weights of good programs by the learning rate
    """

    def _init_weights(self, labeled_demo):
        weight = float(self._config.weight_init)
        for i in xrange(len(labeled_demo)):
            for weighted_program in labeled_demo.programs(i):
                weighted_program.set_weight(weight)
        labeled_demo.initialize_critics(None)

    def compute_program_probs(self, weighted_programs):
        weights = [p.weight for p in weighted_programs]
        return [w / sum(weights) for w in weights]

    def _update_weights(self, demo_player, episode):
        reward = episode.discounted_return(-1, 1.0)
        learning_rate = self._config.learning_rate
        if reward > 0:
            for weighted_program in demo_player.selected_programs:
                weighted_program.incr_weight(learning_rate)


class SoftmaxProgramPolicy(ProgramPolicy):
    """Program policy with softmax weighting scheme.

    - Initialize the weights so that the most precise programs
        get positive weight, while others get 0.
        (Roughly equivalent to MML regularized with infty norm = clipping)
    - Sample the program proportionally to the softmax on the weight
    - Update: gradient update

    Note: Non-zero weight initialization Cannot be used with oracle policies
          since the execution environments are not defined.
    """
    def _init_weights(self, labeled_demo):
        weight = float(self._config.weight_init)
        for i in xrange(len(labeled_demo)):
            weighted_programs = labeled_demo.programs(i)
            if not weighted_programs:
                continue
            if weight == 0.:
                for weighted_program in weighted_programs:
                    weighted_program.set_weight(0.)
            else:
                # Clipping-based: give high weight to precise programs
                env = ExecutionEnvironment(labeled_demo.state(i))
                num_results = []
                for weighted_program in weighted_programs:
                    program = weighted_program.program
                    if program is None:  # Skip is good
                        num_result = 1
                    else:
                        try:
                            num_result = program.execution_paths(env)
                            assert num_result >= 0
                        except ProgramExecutionException as e:
                            num_result = 999
                    num_results.append(num_result)
                # Find programs with minimal number of matches
                min_result = min(num_results)
                assert len(weighted_programs) == len(num_results)
                for program, num_result in zip(weighted_programs, num_results):
                    if num_result == min_result:
                        program.set_weight(weight)
                    else:
                        program.set_weight(0.)

            # TODO: This prunes programs randomly after choosing the most
            # restrictive
            pruned_programs = sorted(
                weighted_programs, reverse=True,
                key=lambda x: x.weight)[:self._config.max_programs]
            labeled_demo.set_programs(i, pruned_programs)
        labeled_demo.initialize_critics(float(self._config.init_v))

    def compute_program_probs(self, weighted_programs):
        # Softmax
        assert len(weighted_programs) > 0
        stuff = np.array([p.weight for p in weighted_programs])
        stuff = np.exp(stuff - np.max(stuff))
        return stuff / np.sum(stuff)

    def _update_weights(self, demo_player, episode):
        labeled_demo = demo_player.demo
        learning_rate = self._config.learning_rate
        alpha = self._config.alpha

        episode_timestep = 0
        assert len(demo_player.trajectory_cursors) == \
                len(demo_player.candidate_programs) == \
                len(demo_player.selected_programs)
        for cursor, selected, candidate_programs in zip(
                demo_player.trajectory_cursors,
                demo_player.selected_programs,
                demo_player.candidate_programs):
            # Skip tokens that terminate
            if isinstance(selected.program, TerminateToken):
                continue

            reward = episode.discounted_return(episode_timestep, 1.0)
            if selected.program is not None:
                action = episode[episode_timestep].action
                curr_state = episode[episode_timestep].state
                a_given_p = demo_player.consistent_programs(
                        action, curr_state, cursor)
                episode_timestep += 1
            else:  # Skip action
                a_given_p = {selected: 1.}  # Skips are deterministic

            program_probs = dict(zip(
                candidate_programs,
                self.compute_program_probs(candidate_programs)))
            action_prob = sum(
                a_given_p[program] * program_probs[program]
                for program in a_given_p)
            action_prob = max(action_prob, 0.00001)  # Clip to avoid numerical issues

            # No discount for now
            # Grad: sum(p(a | prog) * pi(prog) / pi(action) grad log pi(prog))
            # sum is over programs
            for candidate_program in candidate_programs:
                program_prob = program_probs[candidate_program]
                # TODO: Better name
                weight = (a_given_p.get(candidate_program, 0.) *
                          program_prob) / action_prob
                g = weight - program_prob
                returns = reward - labeled_demo.critics[cursor]
                candidate_program.incr_weight(
                        learning_rate * g * returns)

            # Update critic
            labeled_demo.set_critic(cursor,
                (1 - alpha) * labeled_demo.critics[cursor] +
                alpha * reward)

        # Reached end of the episde or the last action terminates
        assert isinstance(episode[-1].action, MiniWoBTerminate) or \
                episode_timestep == len(episode)

################################################

class DemoPlayer(object):
    """Wraps a demo, execution env, and cursor inside of a demonstration.

    Args:
        demo (LabeledDemonstration)
        policy (ProgramPolicy)
        test (bool)
    """
    def __init__(self, demo, policy, test=False):
        self._demo = demo
        self._policy = policy
        self._test = test
        self._env = None
        self._cursor = 0
        # list[int]
        self._trajectory_cursors = []
        # list[WeightedProgram]
        self._selected_programs = []
        # list[list[WeightedProgram]]
        self._candidate_programs = []

    def next_action(self, state):
        """Returns a next sampled action from following this demonstration.
        If demonstration is already played through, returns FAIL.

        Args:
            state (MiniWoBState): the current state

        Returns:
            action (MiniWoBAction)
        """
        # Update environment
        if self._env is None:
            self._env = ExecutionEnvironment(state)
        else:
            self._env.observe(state)

        # Greedy: choose the best action that executes
        if self._test:
            # NOTE: selected_programs and candidate_programs are not updated
            # because you should not be taking gradient steps on test.
            action, new_cursor = self._get_best_action(state, self._cursor)
            self._cursor = new_cursor
            return action
        else:
            # Sample until you get a concrete action
            justifications = []
            while True:
                selected_w_program = self._sample_program(state, self._cursor)

                # Update book-keeping
                weighted_programs, probs = self._programs_and_probs(
                        self._cursor)

                if len(weighted_programs) > 0:
                    self._trajectory_cursors.append(self._cursor)
                    self._selected_programs.append(selected_w_program)
                    self._candidate_programs.append(weighted_programs)
                    state_value = self._demo.critics[self._cursor]
                else:  # Sampled action is a terminate
                    state_value = None
                self._cursor += selected_w_program.state_incr

                program = selected_w_program.program
                if program is None:  # Skip action
                    justifications.append(DemoJustification(
                        weighted_programs, probs, selected_w_program,
                        ElementSet.EMPTY, state_value))
                else:  # Regular weighted program
                    elem_set = ElementSet.EMPTY
                    try:
                        action = program.execute(self._env)
                        elem_set = program.possible_elements(self._env)
                    except ProgramExecutionException as e:
                        logging.info("DemoPlayer: %s", e)
                        action = MiniWoBTerminate()
                    justifications.append(DemoJustification(
                        weighted_programs, probs, selected_w_program,
                        elem_set, state_value))
                    action.justification = DemoJustificationList(
                            justifications)
                    return action

    # TODO: Define a SkipToken?
    def _sample_program(self, state, cursor):
        """Returns a WeightedProgram sampled at the current cursor. The program
        in the WeightedProgram may be None, indicating a skip action.

        Args:
            state (MiniWoBState): concrete state
            cursor (int): index of the current demo state
        Returns:
            WeightedProgram: If the WeightedProgram is None, no programs
            were available for sampling.
        """
        weighted_programs, probs = self._programs_and_probs(cursor)
        if not weighted_programs:  # No programs available for sampling.
            return WeightedProgram(TerminateToken(), 0.)

        weighted_program = np.random.choice(weighted_programs, p=probs)
        return weighted_program

    def _get_best_action(self, state, cursor):
        """Execute the highest scoring program that executes to produce an
        action.

        The justification for the action includes zero or more justifications
        for skip actions, which just advance the cursor.

        Args:
            state (MiniWoBState): concrete state
            cursor (int): index of the current demo state
        Returns:
            action (ProgramAction)
            new_cursor (int): the new cursor position
        """
        def helper(state, cursor, justifications):
            """Returns action, new cursor position keeping track of
            justifications in a list.
            """
            weighted_programs, probs = self._programs_and_probs(cursor)
            assert len(weighted_programs) == len(probs)
            ranked = sorted(zip(weighted_programs, probs),
                            key=lambda x: x[1], reverse=True)

            state_value = self._demo.critics[self._cursor] \
                if len(weighted_programs) > 0 else None
            for weighted_program, prob in ranked:
                program = weighted_program.program
                if program is not None:  # Regular program
                    # See if the program executes
                    try:
                        action = weighted_program.program.execute(self._env)
                    except ProgramExecutionException as e:
                        logging.info("DemoPlayer: %s", e)
                        continue

                    new_cursor = cursor + weighted_program.state_incr
                    # Compute justification
                    element_set = program.possible_elements(self._env)
                    justifications.append(DemoJustification(
                        weighted_programs, probs, weighted_program,
                        element_set, state_value))
                    action.justification = DemoJustificationList(
                            justifications)
                    return action, new_cursor
                else:  # Skip edge
                    new_cursor = cursor + weighted_program.state_incr
                    # Compute justification
                    justifications.append(DemoJustification(
                        weighted_programs, probs, weighted_program,
                        ElementSet.EMPTY, state_value))
                    return helper(state, new_cursor, justifications)
            action = MiniWoBTerminate()
            justifications.append(DemoJustification(
                weighted_programs, probs, None, ElementSet.EMPTY,
                state_value))
            action.justification = DemoJustificationList(justifications)
            return action, cursor
        return helper(state, cursor, [])

    def _programs_and_probs(self, cursor):
        """Returns three parallel lists of weighted programs and their
        probabilities at the current cursor.

        Args:
            cursor (int)

        Returns:
            list[WeightedProgram]
            list[float]
        """
        # Past the end of the demo
        if cursor >= len(self._demo):
            return [], []

        weighted_programs = self._demo.programs(cursor)
        if not weighted_programs:
            return [], []
        probs = self._policy.compute_program_probs(weighted_programs)
        return weighted_programs, probs

    def consistent_programs(self, action, state, timestep):
        """Returns a list of WeightedPrograms that are consistent with the
        action at (state, timestep).

        Args:
            action (MiniWoBAction):
            state (MiniWoBState)
            timestep (int)

        Returns:
            a_given_p ({WeightedProgram: float}): the keys are all the
                consistent programs. the probability of the action given the
                consistent program
        """
        assert timestep < len(self._demo)
        w_programs = self._demo.programs(timestep)
        env = ExecutionEnvironment(state)
        a_given_p = {}
        for w_program in w_programs:
            program = w_program.program
            if program is None:  # Skip skip actions
                continue
            # Non-executable programs are consistent with MiniWoBTerminate
            if isinstance(action, MiniWoBTerminate):
                try:
                    num_execution_paths = program.execution_paths(env)
                    if num_execution_paths == 0:
                        a_given_p[w_program] = 1.
                except ProgramExecutionException as e:
                    a_given_p[w_program] = 1.
            else:  # Regular action was played
                try:  # Sometimes programs cannot execute
                    if program is not None and program.consistent(
                            env, action):
                        num_execution_paths = program.execution_paths(env)
                        a_given_p[w_program] = 1. / num_execution_paths
                except ProgramExecutionException as e:
                    logging.info(
                        "consistent_programs({}, {}, {}, {}): {}".format(
                            self, action, state, timestep, e))
        return a_given_p

    @property
    def demo(self):
        """Returns the LabeledDemonstration object."""
        return self._demo

    @property
    def trajectory_cursors(self):
        """Returns the list[int] of cursors at each selected program."""
        return self._trajectory_cursors

    @property
    def selected_programs(self):
        """Returns the list[WeightedPrograms] that were played in order."""
        return self._selected_programs

    @property
    def candidate_programs(self):
        """Returns the list[list[WeightedPrograms]] of candidate programs."""
        return self._candidate_programs

    @property
    def fields(self):
        """Returns the Fields associated with this demo."""
        return self._demo.fields


class DemoJustification(Justification):
    def __init__(self, weighted_programs, probs, selected,
                 element_set, state_value):
        """
        Args:
            programs (list[WeightedProgram])
            probs (list[float])
            selected (WeightedProgram)
            element_set (ElementSet): execution result of selected (or Empty)
            state_value (float | None): estimated value of the state
        """
        self.weighted_programs = weighted_programs
        self.program_probs = sorted(
            zip(weighted_programs, probs), key=lambda x: x[1], reverse=True)
        self.selected = selected
        self.element_set = element_set
        self.state_value = state_value

    def to_json_dict(self):
        d = OrderedDict()
        d['probs'] = {
            str(prog): round(prob, 3) for prog, prob in self.program_probs}
        d['selected'] = str(self.selected)
        d['element_set'] = [str(elem) for elem in self.element_set.elements]
        d['state_value'] = round(self.state_value, 3) \
                if self.state_value is not None else None
        return d

    def dumps(self):
        items = []
        for program, prob in self.program_probs:
            select_marker = '>' if program == self.selected else ' '
            s = '{} [{:.3f}] {}'.format(select_marker, prob, program)
            items.append(s)
        probs_str = '\n'.join(items)
        elems_str = '\n'.join(str(elem) for elem in self.element_set.elements)
        state_value = None if self.state_value is None \
                else round(self.state_value, 3)

        return 'state value: {}\nprogram probs:\n{}\nelement set:\n{}'.format(
                state_value, indent(probs_str),
                indent(elems_str))


class DemoJustificationList(Justification):
    def __init__(self, justifications):
        self._justifications = justifications

    def to_json_dict(self):
        d = OrderedDict()
        d['justifications'] = [x.to_json_dict() for x in self._justifications]
        return d

    def dumps(self):
        items = []
        for i, x in enumerate(self._justifications):
            items.append('step {} / {}:'.format(i + 1, len(self._justifications)))
            items.append(indent(x.dumps()))
        return '\n'.join(items)
