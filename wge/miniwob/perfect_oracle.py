import abc
import logging
import wge.miniwob.action as A
from wge.rl import Policy


def get_perfect_oracle(subdomain, config):
    if "email" in subdomain:
        return PerfectEmailOracle()
    elif "choose-date" in subdomain:
        return ChooseDateOracle()
    else:
        raise ValueError("No perfect oracle policy for {}".format(subdomain))


class PerfectOracle(Policy):
    def __init__(self):
        super(PerfectOracle, self).__init__()
        self.end_episodes()

    # TODO: Refactor to be like ProgramPolicy?
    def act(self, states, test=False):
        if self._start_over:
            self._start_over = False
            self._players = [self._get_player(state) for state in states]

        actions = []
        for state, player in zip(states, self._players):
            if state is None:
                actions.append(None)
            else:
                action = player.next_action(state)
                actions.append(action)
        return actions

    @abc.abstractmethod
    def _get_player(self, state):
        """Returns an ActionPlayer for each state (list[ActionPlayer])"""
        raise NotImplementedError()

    def end_episodes(self):
        self._start_over = True

    def score_actions(self, states, test=False):
        raise NotImplementedError("Do not call this function")

    def update_from_replay_buffer(self, replay, gamma, take_grad_step):
        raise NotImplementedError("You should not call this.")

    def update_from_episodes(self, episodes, gamma, take_grad_step):
        pass

    @property
    def has_attention(self):
        return False


class ActionPlayer(object):
    """For each state, defines the next action."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def next_action(self, state):
        raise NotImplementedError()


def find_element_by_ref(ref, elements):
    for dom_element in elements:
        if dom_element.ref == ref:
            return dom_element
    raise ValueError("Invalid ref: {}".format(ref))


def find_element_by_text(text, elements):
    for dom_element in elements:
        if dom_element.text == text:
            return dom_element
    raise ValueError("Invalid text: {}".format(text))


class RecipePlayer(ActionPlayer):
    """Takes a list of Actions and plays them in order.

    Args:
        actions (list[Action]): either an Action or (ref, None) for click or
            (ref, string) for type
    """
    def __init__(self, actions):
        self._actions = actions
        self._cursor = 0

    @property
    def done(self):
        return self.cursor >= len(self._actions)

    @property
    def cursor(self):
        return self._cursor

    def next_action(self, state):
        action = self._actions[self.cursor]
        if not isinstance(action, A.MiniWoBAction):
            ref, text = action
            element = find_element_by_ref(ref, state.dom_elements)
            if element is None:
                return A.MiniWoBTerminate()

            if text is None:
                action = A.MiniWoBElementClick(element)
            else:
                action = A.MiniWoBFocusAndType(element, text)
        self._cursor += 1
        return action


class PerfectEmailOracle(PerfectOracle):
    def __init__(self):
        super(PerfectEmailOracle, self).__init__()
        self.end_episodes()

    # TODO: Refactor to be like ProgramPolicy?
    def act(self, states, test=False):
        if self._start_over:
            self._start_over = False
            self._players = [self._get_player(state) for state in states]

        actions = []
        for state, player in zip(states, self._players):
            if state is None:
                actions.append(None)
            else:
                action = player.next_action(state)
                actions.append(action)
        return actions

    def _get_player(self, state):
        task = state.fields["task"]
        if task == "star":
            return EmailStarPlayer(state)
        elif task == "reply":
            return EmailReplyPlayer(state)
        elif task == "delete":
            return EmailTrashPlayer(state)
        elif task == "forward":
            return EmailForwardPlayer(state)
        else:
            raise ValueError("Currently not supported")

    def end_episodes(self):
        self._start_over = True

    def score_actions(self, states, test=False):
        raise NotImplementedError("Do not call this function")

    def update_from_replay_buffer(self, replay, gamma, take_grad_step):
        raise ValueError("You should not call this.")

    def update_from_episodes(self, episodes, gamma, take_grad_step):
        pass

    @property
    def has_attention(self):
        return False


EMAIL_SENDER_REFS = [10, 19, 28]
EMAIL_STAR_REFS = [16, 25, 34]
EMAIL_TRASH_REFS = [15, 24, 33]
EMAIL_REPLY_REF = 51
EMAIL_FORWARD_REF = 54
EMAIL_BODY_REF = 66
EMAIL_TO_REF = 62
EMAIL_SEND_REF = 58


class EmailReplyPlayer(RecipePlayer):
    def __init__(self, state):
        fields = state.fields
        by = [element for element in state.dom_elements
              if element.text == fields["by"] and
              element.ref in EMAIL_SENDER_REFS]
        by_action = A.MiniWoBElementClick(by[0])
        reply_action = (EMAIL_REPLY_REF, None)
        type_action = (EMAIL_BODY_REF, fields["message"])
        send_action = (EMAIL_SEND_REF, None)
        actions = [by_action, reply_action, type_action, send_action]
        super(EmailReplyPlayer, self).__init__(actions)


class EmailStarPlayer(RecipePlayer):
    def __init__(self, state):
        fields = state.fields
        by = [element for element in state.dom_elements
              if element.text == fields["by"] and
              element.ref in EMAIL_SENDER_REFS]
        by = by[0]
        for sender_ref, star_ref in zip(EMAIL_SENDER_REFS, EMAIL_STAR_REFS):
            if by.ref == sender_ref:
                star = star_ref
                break
        actions = [(star, None)]
        super(EmailStarPlayer, self).__init__(actions)


class EmailTrashPlayer(RecipePlayer):
    def __init__(self, state):
        fields = state.fields
        by = [element for element in state.dom_elements
              if element.text == fields["by"] and
              element.ref in EMAIL_SENDER_REFS]
        by = by[0]
        for sender_ref, trash_ref in zip(EMAIL_SENDER_REFS, EMAIL_TRASH_REFS):
            if by.ref == sender_ref:
                trash = trash_ref
                break
        actions = [(trash, None)]
        super(EmailTrashPlayer, self).__init__(actions)


class EmailForwardPlayer(RecipePlayer):
    def __init__(self, state):
        fields = state.fields
        by = [element for element in state.dom_elements
              if element.text == fields["by"] and
              element.ref in EMAIL_SENDER_REFS]
        by_action = A.MiniWoBElementClick(by[0])
        forward_action = (EMAIL_FORWARD_REF, None)
        type_action = (EMAIL_TO_REF, fields["to"])
        send_action = (EMAIL_SEND_REF, None)
        actions = [by_action, forward_action, type_action, send_action]
        super(EmailForwardPlayer, self).__init__(actions)


class ChooseDateOracle(PerfectOracle):
    def _get_player(self, state):
        return ChooseDatePlayer()


# Pseudo-enum
class ChooseDatePhase(object):
    OPEN_DATEPICKER = 0
    CHOOSING_DATE = 1
    SUBMIT = 2


class ChooseDatePlayer(ActionPlayer):
    """Defines the correct actions for all of the given months in choose-date.

    Args:
        months (set(int) | None): list of months, which this is able to
        reach. If None, then can reach all months. For excluded months,
        terminates immediately.
    """
    DATE_PICKER_REF = 5
    SUBMIT_REF = 6
    def __init__(self, months=None):
        if months is None:
            months = set(xrange(1, 13))
        self._months = months
        self._phase = ChooseDatePhase.OPEN_DATEPICKER

    def next_action(self, state):
        target_month = int(state.fields["month"])
        if target_month not in self._months:
            return A.MiniWoBTerminate()

        if self._phase == ChooseDatePhase.OPEN_DATEPICKER:
            elem = find_element_by_ref(
                    self.DATE_PICKER_REF, state.dom_elements)
            self._phase = ChooseDatePhase.CHOOSING_DATE
            return A.MiniWoBElementClick(elem)
        elif self._phase == ChooseDatePhase.CHOOSING_DATE:
            current_month = self._get_current_month(state)
            if target_month > current_month:
                next_elem = find_element_by_text("Next", state.dom_elements)
                return A.MiniWoBElementClick(next_elem)
            elif target_month < current_month:
                prev_elem = find_element_by_text("Prev", state.dom_elements)
                return A.MiniWoBElementClick(prev_elem)
            else:
                day_elem = find_element_by_text(
                        state.fields["day"], state.dom_elements)
                self._phase = ChooseDatePhase.SUBMIT
                return A.MiniWoBElementClick(day_elem)
        elif self._phase == ChooseDatePhase.SUBMIT:
            elem = find_element_by_ref(
                    self.SUBMIT_REF, state.dom_elements)
            return A.MiniWoBElementClick(elem)
        else:
            raise ValueError("Not a valid phase")

    def _get_current_month(self, state):
        def str_to_int(month):
            months = ["January", "February", "March", "April", "May", "June",
                      "July", "August", "September", "October", "November",
                      "December"]
            d = dict(zip(months, xrange(1, 13)))
            return d[month]

        for elem in state.dom_elements:
            if elem.classes == "ui-datepicker-month":
                month = str_to_int(elem.text)
                return month
        raise ValueError("No month was found")
