import collections
import gzip
import json
import logging
import os
import sys
import traceback

from codecs import open
from tqdm import tqdm
from wge import data
from wge.miniwob.action import MiniWoBAction, \
        MiniWoBElementClick, MiniWoBFocusAndType
from wge.miniwob.fields import get_field_extractor, Fields
from wge.rl import Episode, Experience
from wge.miniwob.state import MiniWoBState, DOMElement


def load_demonstrations(task, demo_set_dir, parser='original', logfile=None,
                        min_raw_reward=1.):
    """Loads all demonstrations for the task into list[Episode].

    Args:
        task (string): the name of the task
        demo_set_dir (string): path relative to $RL_DEMO_DIR.
        parser (string): demonstration parser to use
        logfile (file): log the demo parsing steps to this file
        min_raw_reward (float): only use demos with raw reward of at least this

    Returns:
        list[EpisodeGraph]
    """
    demo_dir = os.environ['RL_DEMO_DIR']

    # realpath to resolve symlinks
    task_dir = os.path.realpath(os.path.join(demo_dir, demo_set_dir, task))

    print 'LOADING DEMOS FROM: {}'.format(task_dir)

    try:
        filenames = os.listdir(task_dir)
    except OSError:
        # folder doesn't exist, so no files to load
        filenames = []

    # filter out hidden files
    filenames = [name for name in filenames if name[0] != '.']

    episode_graphs = []
    for filename in tqdm(filenames, desc="Loading demonstrations"):
        filename = os.path.join(task_dir, filename)
        episode_graph = load_demonstration(task, filename, parser, logfile)
        if episode_graph:
            if episode_graph.raw_reward < min_raw_reward:
                logging.warn('Discarding %s since raw reward is %f < %f',
                        filename, episode_graph.raw_reward, min_raw_reward)
            else:
                episode_graphs.append(episode_graph)
    return episode_graphs


def load_demonstration(task, filename, parser, logfile=None):
    """Reads the Episode from filename.

    Args:
        task (string): the task
        filename (string): json file or gzipped json file with a demonstration
        parser (string): demonstration parser to use
        logfile (file): log the demo parsing steps to this file

    Returns:
        EpisodeGraph
    """
    if logfile:
        print >> logfile, '#' * 40
        print >> logfile, 'Reading from {} (parser={})'.format(filename, parser)
    field_extractor = get_field_extractor(task)
    opener = gzip.open if filename.endswith('.gz') else open
    with opener(filename, "r") as f:
        contents = json.load(f)
    try:
        return EpisodeGraph(contents, field_extractor, parser, logfile)
    except Exception as e:
        logging.warning('Cannot load from %s: %s', filename, e, exc_info=True)
        if logfile:
            traceback.print_exc(file=logfile)
        return None


################################################

class ActionEdge(
        collections.namedtuple('ActionEdge', ['action', 'start', 'end'])):
    """Defines an edge in graph of states, where the action moves from the
    start index to the end index of states.

    Args:
        action (MiniWoBAction): the action
        start (int): start index
        end (int): end index
    """
    pass


class StateVertex(
        collections.namedtuple("StateVertex", ["state", "action_edges"])):
    """Consists of a state and the edges (actions) that go to other states.

    Attributes:
        state (MiniWoBState)
        action_edges (list[ActionEdge])
    """

    def base_action_edge(self):
        return self.action_edges[0]


class Chunk(
        collections.namedtuple('Chunk', ['action', 'state', 'target', 'args'])):
    """Chunk of events. Used in the chunk-based demo parser."""
    pass


class DummyActionEdge(
        collections.namedtuple('DummyActionEdge', ['chunk', 'start', 'end', 'reason'])):
    """An edge that does not correspond to a real action.
    Used during graph construction.

    Args:
        chunk (Chunk)
        start (int): start index
        end (int): end index
        reason (any): Reason why this is not an actual ActionEdge
    """
    pass


class EpisodeGraph(collections.Sequence):
    """Reads a demo file and parses out experiences into a graph, where the
    vertices are states and the edges are actions.

    NOTE: FocusAndTypeTokens will appear as two experiences in the raw file.
    """
    # Code for before timing
    BEFORE = 1
    AFTER = 3

    def __init__(self, raw_demo, field_extractor, parser, logfile=None):
        """Initialize an EpisodeGraph.

        Args:
            raw_demo (dict): the json contents of the demo
            field_extractor (Callable): the field extractor for this subtask
            parser (str): parser for converting raw events into actions
            logfile (file): log the parsing steps to this file
        """
        self._raw_states = raw_demo['states']
        self._raw_reward = raw_demo['rawReward']
        self._logfile = logfile
        if parser == 'original':
            self._state_vertices = self._parse_raw_demo_original(raw_demo, field_extractor)
        elif parser == 'chunk' or parser == 'chunk-shortcut':
            self._state_vertices = self._parse_raw_demo_chunk(raw_demo, field_extractor,
                    find_shortcuts=(parser == 'chunk-shortcut'))
        else:
            raise ValueError('Unrecognized demo parser: {}'.format(parser))

    def __len__(self):
        """Number of states."""
        return len(self._state_vertices)

    def __getitem__(self, index):
        """Returns the index-th vertex"""
        return self._state_vertices[index]

    @property
    def utterance(self):
        """Returns the starting utterance (unicode)."""
        return self._state_vertices[0].state.utterance

    @property
    def fields(self):
        """Returns the fields extracted from the utterance (Fields)."""
        return self._state_vertices[0].state.fields

    @property
    def raw_reward(self):
        """Returns the raw reward (float)."""
        return self._raw_reward

    def to_experiences(self):
        """Convert the action edges into Experiences (list[Experience]).

        Only include the actions that are in the action space.
        """
        experiences = []
        for i, state_vertex in enumerate(self._state_vertices):
            is_last = (i == len(self._state_vertices) - 1)
            for action_edge in state_vertex.action_edges:
                if self._check_action_edge(state_vertex, action_edge):
                    experiences.append(Experience(
                        state_vertex.state, action_edge.action,
                        self._raw_reward if is_last else 0., {}))
        return experiences

    def _check_action_edge(self, state_vertex, action_edge):
        if isinstance(action_edge, DummyActionEdge) or not action_edge.action:
            return False
        action = action_edge.action
        if isinstance(action, MiniWoBElementClick):
            return action.element.is_leaf
        elif isinstance(action, MiniWoBFocusAndType):
            fields = state_vertex.state.fields
            return action.element.is_leaf and action.text in fields.values
        else:
            raise ValueError('Unrecognized action: {}'.format(action))

    ################################################
    # Helper functions

    def _target(self, dom):
        for elem in dom:
            if elem.targeted:
                return elem
        # Probably clicking on the instruction div
        #logging.warning("No target.")
        return None

    def _value(self, dom, target):
        for elem in dom:
            if elem == target:
                return elem.value
        return None

    def _dom_diff(self, dom1, dom2):
        """Return the differences in DOM trees.

        Args:
            dom1 (DOMElement): Root of the first DOM tree
            dom2 (DOMElement): Root of the second DOM tree
        Returns:
            list[DOMElement]
        """
        return dom1.diff(dom2)

    def _get_leaves(self, dom):
        """Return all leaves of the dom subtree, excluding text nodes.

        Args:
            dom (DOMElement): Root of the subtree
        Returns:
            list[DOMElement]
        """
        leaves = []
        stack = [dom]
        while stack:
            elt = stack.pop()
            if elt.is_leaf:
                if elt.tag != 't':
                    leaves.append(elt)
            else:
                stack.extend(elt.children)
        return leaves

    def _element_with_ref(self, state, ref):
        """Return the element with a given ref, or None if not found"""
        for elt in state.dom_elements:
            if elt.ref == ref:
                return elt

    ################################################

    def _parse_raw_demo_original(self, raw_demo, field_extractor):
        """Takes the raw demo and spits out the relevant states.

        Algorithm: Look at mousedown / keypress events

        Args:
            raw_demo (dict): json contents of demo file
            field_extractor (FieldsExtractor): the fields extractor for this task

        Returns:
            state_vertices (list[StateVertex])
        """
        # Filter out only for keypresses and mousedowns (BEFORE)
        utterance = raw_demo['utterance']
        if 'fields' in raw_demo:
            fields = Fields(raw_demo['fields'])
        else:
            fields = field_extractor(utterance)
        raw_states = raw_demo["states"]
        state_vertices = []
        actions = []
        vertex_number = 0
        for i, raw_state in enumerate(raw_states[1:]):
            raw_action = raw_state["action"]
            if raw_action["timing"] == EpisodeGraph.BEFORE:
                if raw_action["type"] == "mousedown":
                    miniwob_state = MiniWoBState(
                            utterance, fields, raw_state['dom'])
                    target = self._target(miniwob_state.dom_elements)
                    if not target:      # target = yellow instruction box
                        continue
                    click = MiniWoBElementClick(target)
                    state_vertex = StateVertex(
                            miniwob_state, [ActionEdge(
                                click, vertex_number, vertex_number + 1)])
                    state_vertices.append(state_vertex)
                    vertex_number += 1
                elif raw_action["type"] == "keypress":
                    miniwob_state = MiniWoBState(
                            utterance, fields, raw_state['dom'])
                    char = chr(raw_action["keyCode"])
                    target = self._target(miniwob_state.dom_elements)
                    if not target:      # target = yellow instruction box
                        continue
                    type_action = MiniWoBFocusAndType(target, char)
                    state_vertex = StateVertex(
                            miniwob_state, [ActionEdge(
                                type_action, vertex_number, vertex_number + 1)])
                    state_vertices.append(state_vertex)
                    vertex_number += 1

        # Collapse consecutive FocusAndTypes into one
        for i, vertex in enumerate(state_vertices):
            curr_action = vertex.action_edges[0].action
            if not isinstance(curr_action, MiniWoBFocusAndType):
                continue

            aggregated_text = curr_action.text
            while i + 1 < len(state_vertices):
                next_action = state_vertices[i + 1].action_edges[0].action
                if not isinstance(next_action, MiniWoBFocusAndType) or \
                    curr_action.element != next_action.element:
                        break
                aggregated_text += next_action.text
                del next_action
                del state_vertices[i + 1]
            vertex.action_edges[0] = ActionEdge(
                MiniWoBFocusAndType(curr_action.element, aggregated_text), i, i + 1)

        # Collapse Click then FocusAndType into just FocusAndType
        collapsed_state_vertices = []
        for index in xrange(len(state_vertices) - 1):
            curr_action = state_vertices[index].action_edges[0].action
            next_action = state_vertices[index + 1].action_edges[0].action
            if not(isinstance(curr_action, MiniWoBElementClick) and \
                   isinstance(next_action, MiniWoBFocusAndType) and \
                   curr_action.element == next_action.element):
                collapsed_state_vertices.append(state_vertices[index])
        collapsed_state_vertices.append(state_vertices[-1])

        # Correct the edge indices
        for i, state_vertex in enumerate(collapsed_state_vertices):
            state_vertex.action_edges[0] = ActionEdge(
                    state_vertex.action_edges[0].action, i, i + 1)

        return collapsed_state_vertices

    ################################################

    MODIFIERS = {16: 'SHIFT', 17: 'CTRL', 18: 'ALT'}

    def _parse_raw_demo_chunk(self, raw_demo, field_extractor, find_shortcuts=False):
        """Takes the raw demo and spits out the relevant states.

        Algorithm: Consider each chunk of events that express a single action.
        Possible chunks are:
        - click (mousedown mouseup click)
            - double-click is ignored for now
        - drag (mousedown mouseup click, with different coordinates)
        - type (keydown* keypress keyup)
        - hotkey (keydown* keyup, where keyup is not a modifier key)

        Args:
            raw_demo (dict): json contents of demo file
            field_extractor (FieldsExtractor): the fields extractor for this task
            find_shortcuts (bool): whether to also find possible shortcuts
                in the graph. If false, the graph will be a sequence.

        Returns:
            state_vertices (list[StateVertex])
        """
        utterance = raw_demo['utterance']
        if 'fields' in raw_demo:
            fields = Fields(raw_demo['fields'])
        else:
            fields = field_extractor(utterance)
        raw_states = raw_demo['states']

        # Group BEFORE and AFTER
        # Some AFTER are missing due to event propagation being stopped,
        #   in which case we also use BEFORE for AFTER
        raw_state_pairs = []
        current_before = None
        for i, raw_state in enumerate(raw_states[1:]):
            if raw_state['action']['type'] == 'scroll':
                # Skip all scroll actions
                continue
            if raw_state['action']['timing'] == EpisodeGraph.BEFORE:
                if current_before:
                    # Two consecutive BEFOREs
                    logging.warning('state %d is BEFORE without AFTER', i-1)
                    raw_state_pairs.append((current_before, current_before))
                current_before = raw_state
            elif raw_state['action']['timing'] == EpisodeGraph.AFTER:
                if not current_before:
                    # Two consecutive AFTERs
                    logging.warning('state %d is AFTER without BEFORE', i)
                    current_before = raw_state
                raw_state_pairs.append((current_before, raw_state))
                current_before = None
        if current_before:
            # Lingering BEFORE at the end
            logging.warning('state %d is BEFORE without AFTER', i-1)
            raw_state_pairs.append((current_before, current_before))

        if self._logfile:
            print >> self._logfile, 'Utterance:', utterance
            print >> self._logfile, 'Fields:', fields
            print >> self._logfile, '#' * 10, 'PAIRS'
            for i, (s1, s2) in enumerate(raw_state_pairs):
                print >> self._logfile, '@', i, ':', s1['action'], s2['action']

        chunks = self._chunk_events(raw_state_pairs, utterance, fields)
        chunks = self._collapse_type_actions(chunks)

        if self._logfile:
            print >> self._logfile, 'Utterance:', utterance
            print >> self._logfile, 'Fields:', fields
            print >> self._logfile, '#' * 10, 'CHUNKS'
            for i, chunk in enumerate(chunks):
                print >> self._logfile, '@', i, ':', chunk

        # Create base vertices
        state_vertices = []
        for chunk in chunks:
            start, end = len(state_vertices), len(state_vertices) + 1
            if not chunk.target:
                # Probably clicking/dragging on the instruction box
                continue
            if chunk.action == 'click':
                action = MiniWoBElementClick(chunk.target)
                if chunk.target.is_leaf:
                    action_edge = ActionEdge(action, start, end)
                else:
                    action_edge = DummyActionEdge(chunk, start, end, 'nonleaf')
            elif chunk.action == 'type':
                action = MiniWoBFocusAndType(chunk.target, chunk.args)
                if chunk.target.is_leaf:
                    action_edge = ActionEdge(action, start, end)
                else:
                    action_edge = DummyActionEdge(chunk, start, end, 'nonleaf')
            else:
                action_edge = DummyActionEdge(chunk, start, end, 'unknown')
            # If we don't plan to find shortcuts, we cannot have dummy edges
            if not find_shortcuts and isinstance(action_edge, DummyActionEdge):
                continue
            state_vertex = StateVertex(chunk.state, [action_edge])
            state_vertices.append(state_vertex)

        if self._logfile:
            print >> self._logfile, '#' * 10, 'GRAPH'
            for i, v in enumerate(state_vertices):
                print >> self._logfile, '@', i, ':', v.action_edges
                print >> self._logfile, v.state.dom.visualize()

        if find_shortcuts:
            if self._logfile:
                print >> self._logfile, '#' * 10, 'SHORTCUTS'
            self._find_shortcuts(state_vertices)

        # Remove dummy edges
        for i, state_vertex in enumerate(state_vertices):
            state_vertex.action_edges[:] = [e for e in state_vertex.action_edges
                    if not isinstance(e, DummyActionEdge)]
            # To prevent empty states, add skip edges to the next state
            if not state_vertex.action_edges:
                state_vertex.action_edges.append(
                        ActionEdge(None, i, i + 1))

        if self._logfile:
            print >> self._logfile, '#' * 10, 'FINAL'
            print >> self._logfile, 'Utterance:', utterance
            print >> self._logfile, 'Fields:', fields
            for i, v in enumerate(state_vertices):
                print >> self._logfile, '@', i, ':', v.action_edges

        return state_vertices

    def _chunk_events(self, raw_state_pairs, utterance, fields):
        """Find chunks of events that express a single action."""

        chunks = []
        last_mousedown = None
        last_mouseup = None
        last_keydown = None
        # Current modifier keys (shift, ctrl, alt)
        current_modifiers = set()
        # Number of keypresses left to be checked with keydowns
        pending_keypresses = 0

        for i, (raw_state, raw_state_after) in enumerate(raw_state_pairs):
            raw_action = raw_state['action']
            state = MiniWoBState(utterance, fields, raw_state['dom'])
            target = self._target(state.dom_elements)
            t = raw_action['type']
            if t == 'mousedown':
                if last_mousedown:
                    logging.warning('Two consecutive mousedowns @ %d', i)
                    # Click is missing; convert the last mousedown into a click
                    chunks.append(Chunk('click',
                        last_mousedown.state, last_mousedown.target, last_mousedown.args))
                    last_mousedown = last_mouseup = None
                coord = (raw_action['x'], raw_action['y'])
                last_mousedown = Chunk('mousedown', state, target, coord)
            elif t == 'mouseup':
                assert last_mousedown, 'Cannot have mouseup without mousedown @ {}'.format(i)
                assert not last_mouseup, 'Two consecutive mouseups @ {}'.format(i)
                coord = (raw_action['x'], raw_action['y'])
                last_mouseup = Chunk('mouseup', state, target, coord)
            elif t == 'click':
                if last_mouseup:
                    # TODO: Currently dragging is interpreted as clicking
                    chunks.append(Chunk('click',
                        last_mousedown.state, last_mousedown.target, last_mousedown.args))
                    last_mousedown = last_mouseup = None
                else:
                    # Spurious click event from <label> tag
                    pass
            elif t == 'dblclick':
                # dblclick is ignored (two clicks are already in the chunk list)
                continue
            elif t == 'keydown':
                keycode = raw_action['keyCode']
                if keycode in EpisodeGraph.MODIFIERS:
                    current_modifiers.add(keycode)
                else:
                    last_keydown = Chunk('keydown', state, target,
                        sorted(list(current_modifiers) + [keycode]))

            elif t == 'keyup':
                keycode = raw_action['keyCode']
                if keycode in EpisodeGraph.MODIFIERS:
                    assert keycode in current_modifiers,\
                            'keyup on modifier without keydown @ {}'.format(i)
                    current_modifiers.remove(keycode)
                elif pending_keypresses:
                    pending_keypresses -= 1
                else:
                    assert last_keydown,\
                            'keyup without keydown @ {}'.format(i)
                    # Hotkey
                    state_after = MiniWoBState(utterance, fields, raw_state_after['dom'])
                    chunk = self._resolve_hotkey(
                            last_keydown.state, target, last_keydown.args, state_after)
                    if chunk:
                        chunks.append(chunk)
            elif t == 'keypress':
                char = unichr(raw_action['charCode'])
                chunks.append(Chunk('type',
                    last_keydown.state, last_keydown.target, char))
                pending_keypresses += 1
            else:
                raise ValueError('Unknown action type: {}'.format(t))

        return chunks

    def _resolve_hotkey(self, state, target, keys, state_after):
        """Interpret a hotkey as a simple equivalent action.

        Args:
            state (MiniWoBState)
            target (DOMElement or None)
            keys (list[int]): List of keycodes
            state_after (MiniWoBState)
        Returns:
            Chunk or None
        """
        # Check for common keys
        if keys == [17, 86]:
            # Ctrl + V: Convert to type
            new_value = self._value(state_after.dom_elements, target)
            return Chunk('type', state, target, new_value)
        elif keys == [9] or keys == [17, 67] or keys == [20]:
            # Ignore: TAB, Ctrl + C, CapsLock
            return None
        else:
            # Unknown hotkey sequence
            return Chunk('hotkey', state, target, keys)

    def _collapse_type_actions(self, chunks):
        """Collapse consecutive type actions.
        Also collapse the click that focuses on the element being typed in.

        Args:
            chunks (list[Chunk])
        Returns:
            list[Chunk]
        """
        # Collapse type actions
        collapsed = []
        last_type = None
        for i, chunk in enumerate(chunks):
            if (i + 1 < len(chunks)
                    and chunk.action == 'click'
                    and chunks[i+1].action == 'type'
                    and chunk.target == chunks[i+1].target):
                continue
            elif chunk.action == 'type':
                if not last_type:
                    last_type = chunk
                elif last_type.target == chunk.target:
                    last_type = Chunk('type',
                            last_type.state, last_type.target,
                            last_type.args + chunk.args)
                else:
                    collapsed.append(last_type)
                    last_type = chunk
            elif (chunk.action == 'hotkey'
                    and chunk.args == [8]
                    and last_type and last_type.target == chunk.target):
                # Backspace
                last_type = Chunk('type',
                        last_type.state, last_type.target,
                        last_type.args[:-1])
            else:
                if last_type:
                    collapsed.append(last_type)
                    last_type = None
                collapsed.append(chunk)
        if last_type:
            collapsed.append(last_type)
        return collapsed

    # Maximum length of a skip edge / shortcut edge
    MAX_SHORTCUT_LENGTH = 5
    # If an action involves a non-leaf element, replace it with any leaf descendant,
    # but only if the number of leaves is at most this number
    MAX_LEAVES = 8

    def _find_shortcuts(self, state_vertices):
        """Takes the list of StateVertex and finds ActionEdges between
        non-consecutive state vertices and adds these edges to the passed StateVertexs

        Modifies edges in state_vertices directly

        Args:
            state_vertices (list[StateVertex])
        """
        # Single step:
        for i, vi in enumerate(state_vertices):
            is_last = (i == len(state_vertices) - 1)
            for action_edge in vi.action_edges[:]:
                dom_diff = True if is_last else self._dom_diff(
                        vi.state.dom, state_vertices[i + 1].state.dom)
                if self._logfile and (dom_diff is True or len(dom_diff) <= 5):
                    print >> self._logfile, 'DIFF', i, ':', dom_diff
                if not dom_diff:
                    vi.action_edges.append(ActionEdge(None, i, i + 1))
                elif isinstance(action_edge, DummyActionEdge):
                    chunk = action_edge.chunk
                    if action_edge.reason == 'nonleaf':
                        # Action on non-leaf: Try all leaves instead
                        leaves = self._get_leaves(chunk.target)
                        if len(leaves) <= self.MAX_LEAVES:
                            for leaf in self._get_leaves(chunk.target):
                                if chunk.action == 'click':
                                    action = MiniWoBElementClick(leaf)
                                elif chunk.action == 'type':
                                    action = MiniWoBFocusAndType(leaf, chunk.args)
                                else:
                                    raise ValueError('Invalid nonleaf DummyActionEdge')
                                vi.action_edges.append(ActionEdge(action, i, i + 1))
                    elif not is_last:
                        action = self._find_equivalent_action(
                                dom_diff, vi.state, state_vertices[i + 1].state)
                        if action:
                            vi.action_edges.append(ActionEdge(action, i, i + 1))
        # Multiple steps:
        for i in xrange(len(state_vertices)):
            vi = state_vertices[i]
            for j in xrange(i + 2, min(i + 1 + self.MAX_SHORTCUT_LENGTH, len(state_vertices))):
                vj = state_vertices[j]
                dom_diff = self._dom_diff(vi.state.dom, vj.state.dom)
                if self._logfile and len(dom_diff) <= 5:
                    print >> self._logfile, 'DIFF', i, '->', j, ':', dom_diff
                if not dom_diff:
                    vi.action_edges.append(ActionEdge(None, i, j))
                else:
                    action = self._find_equivalent_action(dom_diff, vi.state, vj.state)
                    if action:
                        vi.action_edges.append(ActionEdge(action, i, j))

    def _find_equivalent_action(self, dom_diff, state_before, state_after):
        """Return a single action that could produce the dom diff.

        Args:
            dom_diff (list[DOMElement])
            state_before (MiniWoBState)
            state_after (MiniWoBState)
        Returns:
            MiniWoBAction or None
        """
        if len(dom_diff) > 1:
            return
        ref = dom_diff[0].ref
        elt_before = self._element_with_ref(state_before, ref)
        elt_after = self._element_with_ref(state_after, ref)
        if not elt_before or not elt_after:
            return
        # Click
        if (elt_before.value == elt_after.value
                and not elt_before.tampered
                and elt_after.tampered):
            return MiniWoBElementClick(elt_before)
        if elt_before.value != elt_after.value:
            if elt_before.tag in ('input_checkbox', 'input_radio'):
                return MiniWoBElementClick(elt_before)
            else:
                return MiniWoBFocusAndType(elt_before, elt_after.value)
