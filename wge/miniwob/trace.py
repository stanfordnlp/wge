from collections import OrderedDict

from gtd.log import indent
from wge.rl import Trace


class MiniWoBEpisodeTrace(Trace):
    def __init__(self, episode):
        self.episode = episode
        self.experience_traces = [
                MiniWoBExperienceTrace(exp) for exp in episode]
        self.utterance = self.episode[0].state.utterance

    def to_json_dict(self):
        d = OrderedDict()
        d['undiscounted_reward'] = self.episode.discounted_return(0, 1.)
        d['reward_reason'] = self.episode[-1].metadata.get('reason')
        d['experiences'] = [
            exp_trace.to_json_dict() for exp_trace in self.experience_traces]
        return d

    def dumps(self):
        exp_strs = []
        for t, trace in enumerate(self.experience_traces):
            exp_str = u'=== time {} ===\n{}'.format(t, trace.dumps())
            exp_strs.append(exp_str)
        s = '\n\n'.join(exp_strs)

        actions_str = ', '.join(str(exp.action) for exp in self.episode)

        return u'Undiscounted reward: {}\nReason: {}\nAction summary: {}\n\n{}'.format(
            self.episode.discounted_return(0, 1.),
            self.episode[-1].metadata.get('reason'),
            actions_str,
            s)


class MiniWoBExperienceTrace(Trace):
    def __init__(self, experience):
        self.experience = experience

        state = experience.state
        if state:
            self._state_str = state.dom.visualize()
        else:
            self._state_str = None

    def to_json_dict(self):
        state, action, _, _ = self.experience
        d = OrderedDict()
        d['utterance'] = state.utterance
        d['fields'] = {k: state.fields[k] for k in state.fields.keys}
        d['state'] = self._state_str
        d['action'] = str(action)
        d['justification'] = action.justification.to_json_dict() if action.justification else None
        d['undiscounted_reward'] = self.experience.undiscounted_reward
        d['metadata'] = self.experience.metadata

        return d

    def dumps(self):
        action = self.experience.action
        return u'utterance: {}\nfields:\n{}\nstate:\n{}\njustification:\n{}\naction:\n{}\nreward: {}\nmetadata: {}'.format(
            self.experience.state.utterance,
            indent(str(self.experience.state.fields)),
            indent(self._state_str),
            indent(action.justification.dumps() if action.justification else 'None'),
            indent(str(action)),
            self.experience.undiscounted_reward,
            self.experience.metadata,
        )
