import abc
import logging
import numpy as np
import random
import wge.miniwob.neighbor as N

from wge.miniwob.action import MiniWoBAction, MiniWoBElementClick, \
        MiniWoBType, MiniWoBFocusAndType, MiniWoBTerminate
from wge.miniwob.distance import line_segment_distance, \
        rectangle_distance
from wge.utils import strip_punctuation, strip_whitespace, word_tokenize, Phrase


class ExecutionEnvironment(object):
    """The execution environment for ProgramTokens. Provides the necessary
    information for the program tokens to execute. Wraps the DOM.

    Args:
        initial_state (MiniWoBState): the initial state
    """
    def __init__(self, initial_state):
        self._dom_elems = initial_state.dom_elements
        self._phrase = initial_state.phrase
        self._fields = initial_state.fields
        self._last = None
        self._expect_state = False  # Expect last then obs
        self._init_cache(initial_state.dom_elements)

    def observe(self, state):
        """Updates the environment based on the next state. Must be
        interleaved with updates to last.

        Args:
            state (MiniWoBState)
        """
        if not self._expect_state:
            raise ValueError("Need to set last first.")

        # Clear the cache when you observe something new
        self._dom_elems = state.dom_elements
        self._phrase = state.phrase
        self._expect_state = False
        self._init_cache(state.dom_elements)

    @property
    def tokens(self):
        """Returns list[unicode]: the tokenized utterance."""
        return list(self._phrase.tokens)

    def detokenize(self, start, end):
        """Returns unicode: the substring corresponds to tokens[start:end]"""
        return self._phrase.detokenize(start, end)

    def elements_by_classes(self, classes):
        """Returns set of DOMElements which have any of the specified classes.

        Args:
            classes (set(string) | None): None matches all elements

        Returns:
            set(DOMElement)
        """
        if classes is None:
            return set(self.elements)

        elements = set()
        for cls in classes:
            elements |= self._cache.get(cls, set())
        return elements

    @property
    def tags(self):
        """Return the set of all tags in the current state set(string)"""
        return set([dom.tag for dom in self.elements])

    @property
    def fields(self):
        """Returns Fields: the fields."""
        return self._fields

    @property
    def buttons(self):
        """Returns ElementSet: all button elements on page."""
        return ElementSet(
                [dom for dom in self._dom_elems if dom.tag == "button"])

    @property
    def text(self):
        """Returns ElementSet: all text elements on page."""
        return ElementSet([dom for dom in self._dom_elems if dom.tag == "t"])

    @property
    def input(self):
        """Returns ElementSet: all input elements on page."""
        return ElementSet(
            [dom for dom in self._dom_elems if "input" in dom.tag or
             dom.tag == "textarea"])

    @property
    def last(self):
        """Returns last modified DOMElement."""
        if self._last is None:
            raise ValueError("Last not set yet.")
        return self._last

    @property
    def elements(self):
        """Returns list[DOMElement] of all elements on page."""
        return self._dom_elems

    @property
    def valid_strings(self):
        """Returns all substrings of DOM element text fields of length <=
        3 (excluding the utterance).

        Returns:
            set(unicode)
        """
        strings = set()
        for dom in self._dom_elems:
            text = dom.text
            if text is not None:
                text = Phrase(strip_punctuation(text))
                for length in xrange(1, 4):
                    for i in xrange(len(text.tokens) - length + 1):
                        strings.add(text.detokenize(i, i + length))
        return strings

    def cache_contains(self, program_str):
        """Returns True if the result of executing the program associated with
        program_str is in the cache.

        Args:
            program_str (string): str(ProgramToken)

        Returns:
            bool
        """
        return program_str in self._cache

    def cache_get(self, program_str):
        """Returns the execution result associated with program_str. Must
        already be in the cache.

        Args:
            program_str (string): str(ProgramToken)

        Returns:
            execution result
        """
        if not self.cache_contains(program_str):
            raise ValueError("{} not cached".format(program_str))
        return self._cache[program_str]

    def cache_set(self, program_str, execution_result):
        """Sets the execution result associated with program_str in the cache.
        Must not already be set

        Args:
            program_str (string): str(ProgramToken)
            execution_result
        """
        if self.cache_contains(program_str):
            raise ValueError("{} already cached".format(program_str))
        self._cache[program_str] = execution_result

    def set_last(self, last):
        """Sets the last property. Must be interleaved with states.

        Args:
            last (DOMElement): the last modified DOM element
        """
        if self._expect_state:
            raise ValueError("Need to observe first.")
        self._last = last
        self._expect_state = True

    def _init_cache(self, dom_elements):
        """Initializes the cache with the classes of all of the DOM elements

        Args:
            dom_elements (list[DOMElement]): list of all DOM elements in the
            current state
        """
        # Cache things on the ExecutionEnv
        # Right now: caching str(ProgramToken) --> ElementSet result
        # Also: cache class name --> set(DOMElement)
        self._cache = {}

        # optimization: class --> elements with that class
        # TODO: There's no reason that this needs to go in cache, should go
        # in a different dict.
        for element in dom_elements:
            for cls in element.classes.split():
                self._cache.setdefault(cls, set()).add(element)


class ProgramToken(object):
    """Base class for all program tokens."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def execute(self, env):
        """Executes this program token on its arguments.

        Args:
            env (ExecutionEnvironment)
        """
        raise NotImplementedError()

    @abc.abstractproperty
    def return_type(self):
        """Returns the return type of executing this ProgramToken."""
        raise NotImplementedError()


class ProgramAction(ProgramToken):
    """A ProgramToken that executes to an Action."""
    @abc.abstractmethod
    def consistent(self, env, action):
        """Returns True if it's possible for this ProgramAction to be
        consistent with the provided action in this environment.

        Args:
            env (ExecutionEnvironment)
            action (MiniWoBAction)

        Returns:
            bool: True if possible to be consistent, otherwise False.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def possible_elements(self, env):
        """Returns the ElementSet of all possible elements that this token
        could click / type. Guaranteed to not throw Exceptions if execute
        succeeds.

        Args:
            env (ExecutionEnvironment)

        Returns:
            ElementSet
        """
        raise NotImplementedError()

    def execution_paths(self, env):
        """Returns the number of possible execution paths of this token on this
        environment.

        Args:
            env (ExecutionEnvironment)

        Returns:
            int
        """
        return len(self.possible_elements(env))

    @property
    def return_type(self):
        return MiniWoBAction


class TerminateToken(ProgramAction):
    """Ignores the environment and returns a Terminate action."""
    def execute(self, env):
        return MiniWoBTerminate()

    def consistent(self, env, action):
        """Only consistent with the Terminate action."""
        return isinstance(action, MiniWoBTerminate)

    def possible_elements(self, env):
        """Does not have any possible elements. Returns empty ElementSet"""
        return ElementSet.EMPTY

    def __str__(self):
        return "TerminateToken"
    __repr__ = __str__


class ClickToken(ProgramAction):
    """Takes a program token that executes to an ElementSet and returns a
    MiniWoBClick.

    Args:
        token (ProgramToken): must execute to an ElementSet
    """
    def __init__(self, token):
        assert token.return_type == ElementSet

        self._parameter = token

    def execute(self, env):
        elements = self._parameter.execute(env)
        element = elements.sample_non_text()

        # Update last
        env.set_last(element)
        return MiniWoBElementClick(element)

    def consistent(self, env, action):
        """Consistent with an action iff the action is an MiniWoBElementClick
        and the element that is clicked is in the ElementSet that the
        arguments execute to.

        NOTE: MiniWoBCoordClick is NOT supported.

        Args:
            env (ExecutionEnvironment)
            action (MiniWoBAction)

        Returns:
            bool
        """
        if isinstance(action, MiniWoBElementClick):
            if not action.element.is_leaf:
                logging.warn(
                        "{} does not click a leaf element.".format(action))

            elements = self._parameter.execute(env)
            for elem in elements.elements:
                if action.ref == elem.ref:
                    return True
            return False
        else:
            return False

    def possible_elements(self, env):
        return self._parameter.execute(env)

    def __str__(self):
        return "Click({})".format(self._parameter)
    __repr__ = __str__


class TypeToken(ProgramAction):
    """Takes a program token that executes to a string and returns
    a MiniWobTypeAction.

    Args:
        token (ProgramToken): must execute to unicode
    """
    def __init__(self, token):
        assert token.return_type == unicode

        self._token = token

    def execute(self, env):
        s = self._token.execute(env)
        return MiniWoBType(s)

    def possible_elements(self, env):
        raise NotImplementedError()

    def consistent(self, env, action):
        if isinstance(action, MiniWoBType):
            s = self._token.execute(env)
            return action.text == s
        else:
            return False


class FocusAndTypeToken(ProgramAction):
    """Takes a program token that executes to an element set and a program
    token that executes to a string and returns a MiniWoBFocusAndType,
    which clicks on an element in the element set and types the string.

    Args:
        elem_token (ProgramToken): executes to ElementSet
        string_token (ProgramToken): executes to unicode
    """
    def __init__(self, elem_token, string_token):
        assert elem_token.return_type == ElementSet
        assert string_token.return_type == unicode
        self._elem_token = elem_token
        self._string_token = string_token

    def execute(self, env):
        elements = self._elem_token.execute(env)
        element = elements.sample_inputable()

        s = self._string_token.execute(env)

        # Update last
        env.set_last(element)
        return MiniWoBFocusAndType(element, s)

    def possible_elements(self, env):
        return self._elem_token.execute(env)

    def consistent(self, env, action):
        if isinstance(action, MiniWoBFocusAndType):
            if not action.element.is_leaf:
                logging.warn(
                        "{} does not click a leaf element.".format(action))

            s = self._string_token.execute(env)
            if s != action.text:
                return False

            elements = self._elem_token.execute(env)
            for elem in elements.elements:
                if action.ref == elem.ref:
                    return True
            return False
        else:
            return False

    def __str__(self):
        return "Type({}, {})".format(self._elem_token, self._string_token)
    __repr__ = __str__


class FocusAndRandomFieldTypeToken(ProgramAction):
    """Randomly types a field. Consistent with all typing actions"""
    def __init__(self, elem_token):
        assert elem_token.return_type == ElementSet
        self._elem_token = elem_token

    def execute(self, env):
        elements = self._elem_token.execute(env)
        element = elements.sample_inputable()

        # Choose a random field
        s = np.random.choice(env.fields.values)

        # Update last
        env.set_last(element)
        return MiniWoBFocusAndType(element, s)

    def possible_elements(self, env):
        return self._elem_token.execute(env)

    def execution_paths(self, env):
        return len(self.possible_elements(env)) * len(env.fields.values)

    def consistent(self, env, action):
        if isinstance(action, MiniWoBFocusAndType):
            if not action.element.is_leaf:
                logging.warn(
                        "{} does not click a leaf element.".format(action))

            elements = self._elem_token.execute(env)
            for elem in elements.elements:
                if action.ref == elem.ref:
                    return True
            return False
        else:
            return False

    def __str__(self):
        return "RandomType({})".format(self._elem_token)
    __repr__ = __str__


class StringToken(ProgramToken):
    """Wrapper around a Python string. Only valid StringTokens are substrings
    of length <= 3 of button text. Executes to the wrapped string.

    Args:
        s (unicode): the wrapped string
    """
    def __init__(self, s):
        assert isinstance(s, unicode)

        self._string = s

    def execute(self, env):
        return self._string

    @property
    def return_type(self):
        return unicode

    def __str__(self):
        return "String({})".format(repr(self._string))
    __repr__ = __str__


class FieldsValueSelectorToken(ProgramToken):
    """Takes an index and executes to the index-th value from Fields
    (fields[index]). Does not support negative indexing. Fields are sorted in
    alphabetical order by keys.

    Args:
        index (int): 0 <= index <= len(fields)
    """
    def __init__(self, index):
        assert 0 <= index
        self._index = index

    def execute(self, env):
        fields = env.fields
        if self._index >= len(fields):
            raise ProgramExecutionException(
                    "fields.values[{}] out of bounds".format(self._index))

        entries = zip(fields.keys, fields.values)
        entries.sort(key=lambda x: x[0])
        _, values = zip(*entries)

        return values[self._index]

    @property
    def return_type(self):
        return unicode

    def __str__(self):
        return "FieldsValueSelector({})".format(self._index)
    __repr__ = __str__


class UtteranceSelectorToken(ProgramToken):
    """Takes a start and end index, which execute to select that
    substring of the utterance. Behaves like utt[start: end].

    Does not support normal Python indexing: in particular, negative indices
    are not supported, start and end must be in bounds, and start < end.

    Args:
        start (int): inclusive
        end (int): exclusive
    """
    def __init__(self, start, end):
        assert 0 <= start
        assert 0 <= end
        assert start < end

        self._start = start
        self._end = end

    def execute(self, env):
        if self._start >= len(env.tokens):
            raise ProgramExecutionException(
                    "utt[{}: {}] start out of bounds".format(
                        self._start, self._end))
        if self._end >= len(env.tokens) + 1:
            raise ProgramExecutionException(
                    "utt[{}: {}] end out of bounds".format(
                        self._start, self._end))
        return env.detokenize(self._start, self._end)

    @property
    def return_type(self):
        return unicode

    def __str__(self):
        return "UtteranceSelector({}, {})".format(self._start, self._end)
    __repr__ = __str__


class ElementSetToken(ProgramToken):
    """Token that executes to an ElementSet.

    Args:
        classes (unicode): optional, space separated list of classes,
            execution excludes all elements whose classes has an empty
            intersection with this classes. None accepts all classes.
    """
    __metaclass__ = abc.ABCMeta
    def __init__(self, classes=None):
        self._classes = set(classes.split()) if classes else None

    def execute(self, env):
        """Performs _execute and caches the result on the environment"""
        if env.cache_contains(str(self)):
            return env.cache_get(str(self))
        result = self._execute(env)
        env.cache_set(str(self), result)
        return result

    @abc.abstractmethod
    def _execute(self, env):
        """Implement self.execute(env)"""
        raise NotImplementedError()

    @property
    def return_type(self):
        return ElementSet

    def _class_match(self, elem):
        """Returns if elem matches self's classes"""
        if self._classes is None:
            return True
        else:
            return len(self._classes.intersection(elem.classes.split())) > 0


class TagToken(ElementSetToken):
    """Executes to the element set of all elements who have the specified
    tag.

    Args:
        tag (string)
    """
    def __init__(self, tag, classes=None):
        super(TagToken, self).__init__(classes)
        self._tag = tag

    def _execute(self, env):
        tag_matches = [elem for elem in env.elements
                       if self._tag == elem.tag and self._class_match(elem)]
        return ElementSet(tag_matches)

    @property
    def return_type(self):
        return ElementSet

    def __str__(self):
        return "TagToken({}, {})".format(self._tag, self._classes)


class InputElementsToken(ProgramToken):
    """Executes to the set of all inputs (ElementSet)."""
    def execute(self, env):
        return env.input

    @property
    def return_type(self):
        return ElementSet

    def __str__(self):
        return "InputElements()"
    __repr__ = __str__


class ButtonsToken(ProgramToken):
    """Executes to the set of all buttons (ElementSet)."""
    def execute(self, env):
        return env.buttons

    @property
    def return_type(self):
        return ElementSet

    def __str__(self):
        return "Buttons()"
    __repr__ = __str__


class TextToken(ProgramToken):
    """Executes to the set of all text elements (ElementSet)."""
    def execute(self, env):
        return env.text

    @property
    def return_type(self):
        return ElementSet

    def __str__(self):
        return "Text()"
    __repr__ = __str__


class LastToken(ProgramToken):
    """Executes to the last modified DOMElement (ElementSet)."""
    def execute(self, env):
        return ElementSet([env.last])

    @property
    def return_type(self):
        return ElementSet

    def __str__(self):
        return "Last()"
    __repr__ = __str__


class DistanceToken(ElementSetToken):
    __metaclass__ = abc.ABCMeta
    """Base class for tokens that map ElementSets to ElementSets by some sort
    of distance metric. Optional classes argument restricts to only elements
    matching that classes arg.

    Args:
        token (ElementSelectorToken): executes to ElementSet
        classes (unicode): optional, space separated list of classes,
            execution excludes all elements whose classes has an empty
            intersection with this classes. None accepts all classes.
    """
    def __init__(self, token, classes=None):
        super(DistanceToken, self).__init__(classes)

        assert token.return_type == ElementSet
        self._token = token

    def _execute(self, env):
        return_elems = set()
        elem_set = self._token.execute(env)
        for elem in elem_set.elements:
            neighbors = self._neighbors(elem, env)
            return_elems |= neighbors
        return ElementSet(list(return_elems))

    def _neighbors(self, elem, env):
        neighbors = set()
        # optimization: only loop through elements that match classes
        neighbor_candidates = env.elements_by_classes(self._classes)
        for neighbor_candidate in neighbor_candidates:
            if self._neighbor_match(elem, neighbor_candidate):
                neighbors.add(neighbor_candidate)
        return neighbors

    @abc.abstractmethod
    def _neighbor_match(self, input_elem, output_elem):
        """Defines if output_elem \in Token(input_elem)

        Args:
            input_elem (DOMElement)
            output_elem (DOMElement)

        Returns:
            bool: True if output_elem \in Token(input_elem)
        """
        raise NotImplementedError()


# TODO: Update Near to be Euclidean distance again -- use SameRow and SameCol
# for row and col stuff
class NearToken(DistanceToken):
    """Executes on a token that produces an ElementSet to produce another
    ElementSet of all the elements within 50px.

    NOTE: Elements are not near themselves.
    """
    def _neighbor_match(self, input_elem, output_elem):
        return N.is_pixel_neighbor(input_elem, output_elem)

    def __str__(self):
        return "Near({}, {})".format(self._token, self._classes)
    __repr__ = __str__


class SameRowToken(DistanceToken):
    """Executes on a token that produces an ElementSet to produce another
    ElementSet of all elements in the same row (horizontal line intersects
    both bounding boxes. Elements are not in the SameRow as themselves.
    """
    def _neighbor_match(self, input_elem, output_elem):
        dist = line_segment_distance(
            input_elem.top, input_elem.top + input_elem.height,
            output_elem.top, output_elem.top + output_elem.height)
        return dist == 0 and input_elem.ref != output_elem.ref

    def __str__(self):
        return "SameRow({}, {})".format(self._token, self._classes)
    __repr__ = __str__


class SameColToken(DistanceToken):
    """Executes on a token that produces an ElementSet to produce another
    ElementSet of all elements in the same col (vertical line intersects
    both bounding boxes. Elements are not in the SameCol as themselves.
    """
    def _neighbor_match(self, input_elem, output_elem):
        dist = line_segment_distance(
            input_elem.left, input_elem.left + input_elem.width,
            output_elem.left, output_elem.left + output_elem.width)
        return dist == 0 and input_elem.ref != output_elem.ref

    def __str__(self):
        return "SameCol({}, {})".format(self._token, self._classes)
    __repr__ = __str__


# TODO: Support LikeID?
class StringMatchToken(ElementSetToken):
    """Takes a unicode token and executes to an ElementSet based on some
    string matching criterion.

    Args:
        token (ProgramToken): executes to unicode
    """
    def __init__(self, token, classes=None):
        super(StringMatchToken, self).__init__(classes)

        assert token.return_type == unicode
        self._token = token

    def _execute(self, env):
        s = self._token.execute(env)
        processed_s = strip_whitespace(strip_punctuation(s))
        matched_elements = set()
        for dom in env.elements:
            if dom.text is not None and self._class_match(dom):
                processed_text = strip_whitespace(strip_punctuation(
                    dom.text))
                if self._string_match(processed_s, processed_text):
                    matched_elements.add(dom)
        return ElementSet(matched_elements)

    @abc.abstractmethod
    def _string_match(self, token_result, dom_text):
        """Returns if the dom_text is a match, given that the token executes
        to token_result.

        Args:
            token_result (unicode): stripped of whitespace and punctuation
            dom_text (unicode): stripped of whitespace and punctuation

        Returns:
            bool
        """
        raise NotImplementedError()


# TODO: Better fuzzy matching
class LikeToken(StringMatchToken):
    """Takes token that executes to unicode and returns ElementSet of
    elements close to the unicode.

    NOTE: Currently only doing exact string match without punctuation and
    whitespace. Now also supports case insensitive substring matching
    (matches if token is substring of text)

    Args:
        token (ProgramToken): executes to unicode
    """
    def _string_match(self, token_result, dom_text):
        return token_result.lower() in dom_text.lower()

    def __str__(self):
        return "Like({}, {})".format(self._token, self._classes)
    __repr__ = __str__


class ExactMatchToken(StringMatchToken):
    """Performs exact string matching."""
    def _string_match(self, token_result, dom_text):
        return token_result.lower() == dom_text.lower()

    def __str__(self):
        return "Exact({}, {})".format(self._token, self._classes)
    __repr__ = __str__


class ProgramExecutionException(Exception):
    """Raise for recoverable program execution errors due to environment."""
    pass


# TODO: Add a prob dist. over the elements.
class ElementSet(object):
    """Set of DOM elements.

    Args:
        elements (list[DOMElement]): the elements
    """
    def __init__(self, elements):
        self._elements = sorted(set(elements), key=lambda e: str(e))
        # TODO(kelvin): find a key that uniquely sorts this list, e.g. `ref`
        # instead of `str`
        # the order needs to be deterministic, to guarantee reproducibility

    def sample_non_text(self):
        """Samples a leaf non text element from the set according to some
        internal probability distribution. (Currently uniform).

        Returns:
            DOMElement
        """
        non_text = [
            element for element in self._elements if element.tag != "t" and
            element.is_leaf]

        if len(non_text) == 0:
            raise ProgramExecutionException("No non-text elements")

        return random.sample(non_text, 1)[0]

    def sample_inputable(self):
        """Samples a leaf input element from the set according to some
        internal probability distribution. (Currently uniform).

        Returns:
            DOMElement
        """
        # TODO: Move all of these into a Tag object
        inputable = [elem for elem in self.elements if
                     ("input" in elem.tag or "textarea" in elem.tag) and
                     elem.is_leaf]

        if len(inputable) == 0:
            raise ProgramExecutionException("No inputable elements")

        return random.sample(inputable, 1)[0]

    @property
    def elements(self):
        return self._elements

    def __len__(self):
        return len(self._elements)

    def __getitem__(self, i):
        return self._elements[i]

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._elements == other._elements
        raise NotImplementedError()

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self == other
        raise NotImplementedError()

    def __hash__(self):
        return hash(tuple(sorted(self._elements)))

    def __str__(self):
        return "ElementSet({})".format(self._elements)
    __repr__ = __str__

ElementSet.EMPTY = ElementSet([])
