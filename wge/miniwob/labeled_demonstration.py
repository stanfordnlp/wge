"""Program lattice."""
import itertools
from codecs import open

from wge.miniwob.action import MiniWoBElementClick, MiniWoBFocusAndType
from wge.miniwob.program import ExecutionEnvironment, \
    ClickToken, StringToken, UtteranceSelectorToken, NearToken, LikeToken, \
    FieldsValueSelectorToken, FocusAndTypeToken, SameRowToken, SameColToken, \
    TagToken, ExactMatchToken, FocusAndRandomFieldTypeToken


class LabeledDemonstration(object):
    """Wrapper around a demonstration Episode, with the programs for each
    action labeled.

    Important private properties:
        _utterance (str): utterance
        _fields (Fields): fields extracted from the utterance
        _programs (list[WeightedProgram]):
            _programs[i] = list of programs starting from state i
        _critics (list[float]): value function for each index
    """
    def __init__(self, utterance, fields, programs, states=None):
        self._utterance = utterance
        self._fields = fields
        self._programs = programs
        self._states = states
        self._critics = None  # Uninitialized

    @classmethod
    def from_episode_graph(cls, episode_graph):
        """Construct a labeled demonstration from an EpisodeGraph.

        Args:
            episode_graph (EpisodeGraph)

        Returns:
            LabeledDemonstration
        """
        programs = [LabeledDemonstration._edges_to_programs(state_vertex)
                for state_vertex in episode_graph]
        states = [state_vertex.state for state_vertex in episode_graph]
        return cls(episode_graph.utterance, episode_graph.fields, programs, states)

    @classmethod
    def from_oracle_programs(cls, oracle_programs, utterance, fields):
        """Constructs a labeled demonstration from programs.

        Args:
            programs (list[list[WeightedProgram]]): programs[i] are the list
                of weighted programs for step i.
            utterance (unicode): the utterance
            fields (Fields): the fields of the utterance

        Returns:
            LabeledDemonstration
        """
        return cls(utterance, fields, oracle_programs)

    def __len__(self):
        return len(self._programs)

    def programs(self, index):
        """Returns a list of WeightedProgram corresponding to the index-th
        action in this demonstration.

        Args:
            index (int)

        Returns:
            list[WeightedProgram]
        """
        return self._programs[index]

    def set_programs(self, index, programs):
        """Sets the WeightedPrograms corresponding to the index-th action in
        this demo.

        Args:
            index (int)
            programs (list[WeightedProgram])
        """
        self._programs[index] = programs

    def state(self, index):
        """Returns the MiniWoBState at the specified index.

        Args:
            index (int)

        Returns:
            MiniWoBState

        Raises:
            ValueError: if the states are not available
        """
        if self._states is None:
            raise ValueError('State is not available in this demonstration')
        return self._states[index]

    @property
    def utterance(self):
        return self._utterance

    @property
    def fields(self):
        return self._fields

    @property
    def critics(self):
        if self._critics is None:
            raise ValueError("Critics not initialized yet!")
        # TODO: This can be changed by the client
        return self._critics

    def initialize_critics(self, value):
        """Initializes all critics to the provided value.

        Args:
            value (float)
        """
        self._critics = [value] * len(self)

    def set_critic(self, index, new_value):
        """Sets critics[index] = new_value"""
        if self._critics is None:
            raise ValueError("Critics not initialized yet!")
        self._critics[index] = new_value

    @staticmethod
    def _edges_to_programs(vertex):
        """Collect ActionEdges originating from the given StateVertex, and list
        all WeightedPrograms that could execute to the actions in those edges.

        Args:
            vertex (StateVertex)

        Returns:
            list[WeightedProgram]
        """
        weighted_programs = []
        env = ExecutionEnvironment(vertex.state)
        
        for action_edge in vertex.action_edges:
            action = action_edge.action
            state_incr = action_edge.end - action_edge.start
            if action is None:
                weighted_programs.append(WeightedProgram(None, 1., state_incr))
                continue

            # All string tokens
            strings = [StringToken(s) for s in env.valid_strings]

            # All fields tokens
            fields = env.fields
            fields_tokens = [
                FieldsValueSelectorToken(i) for i in xrange(len(fields.keys))]
            strings += fields_tokens

            # TODO: Support last. Hard because it depends on the actual exec
            # env.
            element_sets = [TagToken(tag) for tag in env.tags]
            # All of the Like
            element_sets += [
                    LikeToken(string_token) for string_token in strings]
            element_sets += [
                    ExactMatchToken(string_token) for string_token in strings]

            # Max one-level of Near, SameRow, SameCol
            classes = action.element.classes
            distance_programs = [
                NearToken(elem_token, classes) for elem_token in element_sets]
            distance_programs += [
                SameRowToken(elem_token, classes) for elem_token in
                element_sets]
            distance_programs += [
                SameColToken(elem_token, classes) for elem_token in
                element_sets]
            element_sets += distance_programs

            click_actions = [
                ClickToken(element_token) for element_token in element_sets]
            type_actions = [
                FocusAndTypeToken(element_token, string_token) for
                element_token, string_token in
                itertools.product(element_sets, fields_tokens)]
            # Random typing actions
            type_actions += [
                FocusAndRandomFieldTypeToken(element_token) for
                element_token in element_sets]

            if isinstance(action, MiniWoBElementClick):
                consistent_clicks = [
                    WeightedProgram(click, 1., state_incr) for click in
                    click_actions if click.consistent(env, action)]
                weighted_programs.extend(consistent_clicks)
            elif isinstance(action, MiniWoBFocusAndType):
                consistent_types = [
                    WeightedProgram(type_action, 1., state_incr) for type_action in
                    type_actions if type_action.consistent(env, action)]
                weighted_programs.extend(consistent_types)
            else:
                raise ValueError("Action: {} not supported.".format(action))

        return weighted_programs


class WeightedProgram(object):
    """A ProgramAction associated with a weight.

    Args:
        program (ProgramAction)
        weight (float)
        state_incr (int): Increment to the state index after executing the action.
            The default is +1.
    """
    def __init__(self, program, weight, state_incr=1):
        self._program = program
        self._weight = weight
        self._state_incr = state_incr

    @property
    def program(self):
        return self._program

    @property
    def weight(self):
        return self._weight

    @property
    def state_incr(self):
        return self._state_incr

    def set_weight(self, weight):
        self._weight = weight

    def incr_weight(self, incr):
        self._weight += incr

    def __str__(self):
        return "{}{} {:.3f}".format(
                self.program,
                '[{:+}]'.format(self.state_incr) if self.state_incr != 1 else '',
                self.weight)
    __repr__ = __str__


################################################
# Quick Test

def _test():
    import sys
    if len(sys.argv) != 5:
        print >> sys.stderr, 'Usage: {} TASK BASEDIR PARSER LOGFILE'.format(sys.argv[0])
        exit(1)
    from wge.miniwob.demonstrations import load_demonstrations
    assert sys.argv[4].startswith('/tmp/')      # For safety
    with open(sys.argv[4], 'w', 'utf8') as fout:
        episode_graphs = load_demonstrations(sys.argv[1], sys.argv[2], sys.argv[3], fout)
    for episode_graph in episode_graphs:
        print '=' * 40
        for raw_state in episode_graph._raw_states:
            if raw_state['action'] and raw_state['action']['timing'] == 1:
                print raw_state['action']['type'], raw_state['action']
        print '-' * 40
        print 'LENGTH:', len(episode_graph)
        for i, thing in enumerate(episode_graph):
            print i, ':', thing
        print '-' * 40
        labeled_demo = LabeledDemonstration.from_episode_graph(episode_graph)
        for i, thing in enumerate(episode_graph):
            print i, ':', thing
            for program in labeled_demo.programs(i):
                print ' ', program

if __name__ == '__main__':
    _test()
