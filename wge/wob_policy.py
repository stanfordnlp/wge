import abc
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import wge.miniwob.positions as positions
import wge.utils as utils

from collections import namedtuple
from gtd.log import indent
from gtd.ml.torch.attention import Attention, SentinelAttention
from gtd.ml.torch.seq_batch import SequenceBatch
from gtd.ml.torch.utils import GPUVariable, to_numpy
from gtd.utils import as_batches, flatten
from torch.autograd import Variable
from torch.nn import Linear
from wge.embeddings import UtteranceEmbedder, UtteranceVocab
from wge.mask import Mask
from wge.miniwob.action import MiniWoBElementClick, \
    MiniWoBFocusAndType, MiniWoBTerminate
from wge.miniwob.embeddings import HigherOrderDOMElementEmbedder, \
    DOMContextEmbedder, BaseDOMElementEmbedder, DOMAlignments
from wge.miniwob.fields import Fields
from wge.miniwob.state import DOMElementPAD
from wge.miniwob.trace import MiniWoBEpisodeTrace
from wge.replay import ReplayBufferNotReadyException
from wge.rl import Policy, Justification, Episode, Trace, ScoredExperience, ActionScores
from wge.utils import word_tokenize


class MiniWoBPolicy(Policy):
    CLICK = 0
    TYPE = 1
    __metaclass__ = abc.ABCMeta

    @classmethod
    def from_config(cls, config):
        """Initializes policy from config file.

        Args:
            config (Config): must have the following fields (see constructor
            for details)
                - episodes_to_replay
                - scoring_batch_size
                - update_rule
                - baseline
                - use_critic
                - context_embed_dim
                - attn_dim
                - utterance_embedder
                - dom_embedder

        Returns:
            MiniWoBPolicy
        """
        if config.query_type == "structured":
            class_type = StructuredQueryPolicy
        elif config.query_type == "natural-language":
            class_type = NaturalLanguagePolicy
        else:
            raise ValueError("{} does not specify a valid query type".format(
                config.query_type))

        return class_type(
            config.episodes_to_replay, config.scoring_batch_size,
            config.update_rule, config.baseline, config.use_critic,
            config.attn_dim, config.context_embed_dim,
            config.dom_attention_for_state,
            config.utterance_embedder, config.dom_embedder)

    def __init__(self, episodes_to_replay, scoring_batch_size, update_rule,
                 baseline, use_critic, attn_dim, context_embed_dim,
                 dom_attention_for_state, utterance_embedder_config,
                 dom_embedder_config):
        """
        Args:
            episodes_to_replay (int): number of episodes to sample from the
                replay buffer
            scoring_batch_size (int): batch size when scoring episodes
            update_rule (string): determines if updating on whole episode or
                just last experience
            baseline (float): constant subtracted baseline in REINFORCE
            attn_dim (int): dim for all attention modules
            context_embed_dim (int): dim for context
            dom_attention_for_state (bool): whether to attend over the DOM to
                compute state embedding
            utterance_embedder_config (Config): config for utterance embedder
            dom_embedder_config (Config): config for dom embedder
        """
        super(MiniWoBPolicy, self).__init__()
        self._episodes_to_replay = episodes_to_replay
        self._scoring_batch_size = scoring_batch_size
        self._update_rule = update_rule
        self._baseline = baseline
        self._use_critic = use_critic
        self._action_attention = None
        self._dom_attention_for_state = dom_attention_for_state

        assert update_rule in ("use-whole-episode", "only-last-experience")

        self._utterance_embedder = UtteranceEmbedder.from_config(
                utterance_embedder_config)

        query_dim = self._query_embed_dim

        self._base_dom_embedder = BaseDOMElementEmbedder.from_config(
                self._utterance_embedder, dom_embedder_config)
        self._dom_to_field_alignment = DOMAlignments(query_dim)
        self._higher_order_dom_embedder = HigherOrderDOMElementEmbedder(
                self._base_dom_embedder.embed_dim +
                self._dom_to_field_alignment.embed_dim)

        dom_dim = self._higher_order_dom_embedder.embed_dim

        # attends over dom elements
        if self._dom_attention_for_state:
            self._first_dom_attention = Attention(dom_dim, dom_dim, attn_dim)

        # Two heads for second DOM attention
        self._second_dom_attention_heads = nn.ModuleList(
            [Attention(dom_dim, dom_dim + query_dim * 2, attn_dim),
             Attention(dom_dim, dom_dim + query_dim * 2, attn_dim)])

        self._context_embedder = DOMContextEmbedder(
                dom_dim + query_dim * 2, context_embed_dim)
        self._click_or_type_linear = Linear(context_embed_dim, 2)

        sentinel_embed = GPUVariable(torch.zeros(query_dim))

        # Two heads for fields attention
        self._fields_attention_heads = nn.ModuleList(
            [SentinelAttention(query_dim, dom_dim, attn_dim, sentinel_embed),
             SentinelAttention(query_dim, dom_dim, attn_dim, sentinel_embed)])

        # Weights per each of the second DOM attentions
        self._second_dom_attn_head_weights = Linear(query_dim * 2, 2)

        # Linear module for value function
        state_embed_dim = 2 * query_dim + dom_dim
        self._value_function_layer = Linear(state_embed_dim, 1)

    @classmethod
    def _sample(cls, probs):
        """Sample elements.

        Args:
            probs (Variable): of shape (batch_size, seq_len)

        Returns:
            indices (list[int]): list of indices
        """
        # (batch_size, 1)
        sampled_indices = torch.multinomial(probs, 1)
        sampled_indices = torch.squeeze(sampled_indices, 1)

        # back to numpy
        sampled_indices = sampled_indices.data.cpu().numpy()
        return sampled_indices

    @classmethod
    def _greedy_sample(cls, probs):
        _, indices = torch.max(probs, 1)
        indices = torch.squeeze(indices, 1)
        indices = indices.data.cpu().numpy()
        return indices

    @property
    def has_attention(self):
        return True

    @property
    def action_attention(self):
        """Returns the attention over the actions from the previous call to
        act.

        Returns:
            list[np.array or None]: batch of attention
        """
        if self._action_attention is None:
            raise ValueError("act has not been called yet")
        return self._action_attention

    def score_actions(self, states, force_dom_attn=None,
                      force_type_values=None):
        """Score actions.

        See self._score_actions for details.
        """
        mask = Mask(states)
        states = mask.filter(states)
        action_scores_batch = self._score_actions(
                states, force_dom_attn, force_type_values)
        return mask.insert_none(action_scores_batch)

    def _score_actions(self, states, force_dom_attn=None,
                       force_type_values=None):
        """Score actions.

        Args:
            states (list[State])
            force_dom_attn (list[DOMElement]): a batch of DOMElements. If not None,
                forces the second DOM attention to select the specified elements.
            force_type_values (list[unicode | None]): force these values to
                be scored if possible (optional)

        Returns:
            action_scores_batch (list[ActionScores])
        """
        if len(states) == 0:
            return []  # no actions to score

        # ===== Embed entries of the query: (key, value) pairs =====
        # concatenated keys and values of structured query
        query_entries = self._query_entries(states)
        query_embeds = self._query_embeds(states, query_entries)

        # ===== Embed DOM elements =====
        # dom_embeds: SequenceBatch of shape (batch_size, num_elems, dom_dim)
        # dom_elems: list[list[DOMElement]] of shape (batch_size, num_elems)
        #     It is padded with DOMElementPAD objects.
        dom_embeds, dom_elems = self._dom_embeds(states)

        # ===== Embed agent state =====
        # (batch_size, dom_dim)
        dom_embeds_max = SequenceBatch.reduce_max(dom_embeds)

        if self._dom_attention_for_state:
            first_dom_attn_query = dom_embeds_max
            first_dom_attn = self._first_dom_attention(
                    dom_embeds, first_dom_attn_query)
            state_embeds = first_dom_attn.context  # (batch_size, dom_dim)
        else:
            state_embeds = dom_embeds_max

        # ===== Attend over entries of the query =====
        # use both fields attention heads
        fields_attn = [  # List of AttentionOutput objects
            self._fields_attention_heads[0](query_embeds, state_embeds),
            self._fields_attention_heads[1](query_embeds, state_embeds),
        ]

        # (batch_size, attn_dim * 2)
        attended_query_embeds = torch.cat(
            [fields_attn[0].context, fields_attn[1].context], 1)

        # ===== Attend over DOM elements =====
        elem_query = torch.cat([attended_query_embeds, state_embeds], 1)

        # compute state values using elem_query
        state_values = self._value_function_layer(elem_query)  # (batch_size, 1)
        state_values = torch.squeeze(state_values, 1)  # (batch_size,)
        # TODO(kelvin): clean this up

        # two DOM attention heads
        second_dom_attn = [  # contexts have shape (batch_size, dom_dim)
            self._second_dom_attention_heads[0](dom_embeds, elem_query),
            self._second_dom_attention_heads[1](dom_embeds, elem_query)
        ]

        # ===== Compute DOM probs from field weights =====
        dom_head_weights = F.softmax(  # Weight per each head
                self._second_dom_attn_head_weights(attended_query_embeds))

        first_head_weights = torch.index_select(
                dom_head_weights, 1,
                GPUVariable(torch.LongTensor([0]))).expand_as(
                        second_dom_attn[0].weights)
        second_head_weights = torch.index_select(
                dom_head_weights, 1,
                GPUVariable(torch.LongTensor([1]))).expand_as(
                        second_dom_attn[1].weights)

        # DOM probs =
        # dom_head_weights[0] * first_head + dom_head_weights[1] * second_head
        dom_probs = first_head_weights * second_dom_attn[0].weights + \
                    second_head_weights * second_dom_attn[1].weights

        # ===== Decide whether to click or type =====
        HARD_DOM_ATTN = True
        # TODO: Need to fix this for Best First Search
        if HARD_DOM_ATTN:
            # TODO: Bring back the test time flag?
            selector = lambda probs: self._sample(probs)

            if force_dom_attn:
                elem_indices = []
                for batch_idx, force_dom_elem in enumerate(force_dom_attn):
                    refs = [elem.ref for elem in dom_elems[batch_idx]]
                    elem_indices.append(refs.index(force_dom_elem.ref))

                # this selector just ignores probs and returns the indices of
                # the forced elements
                selector = lambda probs: elem_indices

            dom_selection = Selection(selector, dom_probs, candidates=None)
            batch_size, num_dom_elems, dom_dim = dom_embeds.values.size()
            selected_dom_indices = dom_selection.indices
            # (batch_size, 1)
            selected_dom_indices = torch.unsqueeze(selected_dom_indices, 1)
            # (batch_size, 1, 1)
            selected_dom_indices = torch.unsqueeze(selected_dom_indices, 1)
            selected_dom_indices = selected_dom_indices.expand(
                    batch_size, 1, dom_dim)  # (batch_size, 1, dom_dim)

            # (batch_size, 1, dom_dim)
            selected_dom_embeds = torch.gather(
                dom_embeds.values, 1, selected_dom_indices)
            # (batch_size, dom_dim)
            selected_dom_embeds = torch.squeeze(selected_dom_embeds, 1)
        else:
            selected_dom_embeds = torch.cat(
                [second_dom_attn[0].context, second_dom_attn[1].context], 1)

        # (batch_size, context_dim)
        dom_contexts = self._context_embedder(
                selected_dom_embeds, attended_query_embeds)

        # ===== Decide what value to type =====
        type_values, type_value_probs = self._type_values_and_probs(
                states, query_embeds, dom_contexts, force_type_values)

        # (batch_size, 2) (index 0 corresponds to click)
        click_or_type_probs = F.softmax(
                self._click_or_type_linear(dom_contexts))

        action_scores_batch = self._compute_action_scores(
                dom_selection.indices.data.cpu().numpy(), dom_elems, dom_probs,
                click_or_type_probs, type_values, type_value_probs,
                state_values)

        # add justifications
        for batch_idx, action_score in enumerate(action_scores_batch):
            justif = MiniWoBPolicyJustification(
                dom_elements=dom_elems[batch_idx],
                element_probs=dom_probs[batch_idx],
                click_or_type_probs=click_or_type_probs[batch_idx],
                query_entries=query_entries[batch_idx],
                fields_attentions=[fields_attn[0].weights[batch_idx],
                                   fields_attn[1].weights[batch_idx]],
                type_values=type_values[batch_idx],
                type_value_probs=type_value_probs[batch_idx],
                state_value=state_values[batch_idx])
            action_score.justification = justif

        return action_scores_batch

    # TODO: elem_indices is hacked on here to prevent other actions from being scored
    @staticmethod
    def _compute_action_scores(elem_indices, dom_elems, dom_probs,
                               click_or_type_probs, type_values, type_value_probs,
                               state_values):
        """Compute action scores (log probabilities).

        Args:
            dom_elems (list[list[DOMElement]]): a batch of DOMElement lists
                (padded to the same length)
            dom_probs (Variable): of shape (batch_size, max_dom_elems)
            click_or_type_probs (Variable): of shape (batch_size, 2). 0 index
                = click, 1 index = type
            type_values (list[list[unicode]]): a batch of query values (NOT
                padded!)
            type_value_probs (Variable): of shape (batch_size, max_values)
            state_values (Variable): of shape (batch_size,)

        Returns:
            action_scores_batch (list[ActionScores])
        """
        # shape checks
        batch_size = len(dom_elems)
        dim1, max_dom_elems = dom_probs.size()
        dim2, max_type_values = type_value_probs.size()
        dim3, click_or_type_classes = click_or_type_probs.size()
        assert dim1 == batch_size
        assert dim2 == batch_size
        assert dim3 == batch_size
        assert click_or_type_classes == 2
        assert len(type_values) == batch_size

        # check that dom_elems has been appropriately padded
        for elems in dom_elems:
            assert len(elems) == max_dom_elems

        # NOTE: type_values are NOT padded

        action_scores_batch = []
        for batch_idx in range(batch_size):
            action_scores = {}

            click_prob, type_prob = click_or_type_probs[batch_idx]
            # these are scalar Variables

            dom_elems_b, dom_probs_b = dom_elems[batch_idx], dom_probs[batch_idx]
            type_values_b, type_value_probs_b = type_values[batch_idx], type_value_probs[batch_idx]
            assert len(dom_elems_b) == len(dom_probs_b)
            assert len(type_values_b) <= len(type_value_probs_b)

            # TODO: HACK :'(
            chosen_index = elem_indices[batch_idx]
            elem = dom_elems_b[chosen_index]
            elem_prob = dom_probs_b[chosen_index]
            # elem_prob is a scalar Variable: it has size == (1,)

            assert not isinstance(elem, DOMElementPAD)
            assert elem.tag != 't'

            # generate click action
            click_action = MiniWoBElementClick(elem)

            action_scores[click_action] = torch.log(click_prob) + \
                torch.log(elem_prob)

            # generate focus-and-type actions
            for type_value, value_prob in zip(type_values_b, type_value_probs_b):
                # note that zip truncates to the shorter of its two arguments
                type_action = MiniWoBFocusAndType(elem, type_value)
                action_scores[type_action] = torch.log(type_prob) + \
                        torch.log(elem_prob) + torch.log(value_prob)

            state_value = state_values[batch_idx]  # scalar Variable
            action_scores_batch.append(ActionScores(action_scores, state_value))

        return action_scores_batch

    def act(self, states, test=False):
        action_scores_batch = self.score_actions(states)

        actions = []
        for action_scores in action_scores_batch:
            if action_scores is None:
                action = None
            else:
                if test:
                    # action = action_scores.best_action
                    action = action_scores.sample_action()  # always sample, even at test time!
                else:
                    action = action_scores.sample_action()
                action.justification = action_scores.justification  # propagate justification
            actions.append(action)

        # TODO(kelvin): add proper attention visualizer
        self._action_attention = [None for _ in states]

        return actions

    def _pad_elements(self, dom_elems):
        """Takes a batch of dom element lists. Returns the batch with pads so
        that each batch is the same length, and masks.

        Args:
            dom_elems (list[list[DOMElement]]): unpadded batch

        Returns:
            list[list[DOMElement]], Variable[FloatTensor]: batch x num_elems
        """
        # Pad everything to be the same as longest list
        num_elems = max(len(dom_list) for dom_list in dom_elems)
        mask = torch.ones(len(dom_elems), num_elems)
        for dom_list, submask in zip(dom_elems, mask):
            # Avoid empty slice torch errors
            if len(dom_list) < num_elems:
                submask[len(dom_list): num_elems] = 0.
                dom_list.extend(
                    [DOMElementPAD()] * (num_elems - len(dom_list)))

            # TODO: Get rid of these hack.
            # TODO(kelvin): WARNING: this hack also means that we cannot ATTEND to these items
            for i, elem in enumerate(dom_list):
                # never click text elements
                if elem.tag == "t":
                    submask[i] = 0.

        return dom_elems, GPUVariable(mask)

    def _clear_cache(self):
        self._utterance_embedder.clear_cache()

    def _dom_embeds(self, states):
        """Returns the DOM embeddings for a batch of states. Only embeds leaf
        DOM elements, and pads them.

        Args:
            states (list[MiniWoBState])

        Returns:
            dom_embeds (SequenceBatch): batch x num_elems x embed_dim
            dom_elems (list[list[DOMElement]]): of shape (batch_size,
                num_elems). Padded with DOMElementPAD.
        """
        leaf_elems = [
            [elem for elem in state.dom_elements if elem.is_leaf]
            for state in states]
        dom_elems, dom_mask = self._pad_elements(leaf_elems)
        # batch x num_dom_elems x base_dom_embed_dim
        base_dom_embeds = self._base_dom_embedder(dom_elems)
        # batch x num_dom_elems x field_key_embed_dim
        dom_alignment_vectors = self._dom_to_field_alignment(
                dom_elems, self._alignment_fields(states))
        # batch x num_dom_elems x (base + fields embed dim)
        aligned_dom_embeds = torch.cat(
                [base_dom_embeds, dom_alignment_vectors], 2)
        higher_order_dom_embeds = self._higher_order_dom_embedder(
                dom_elems, aligned_dom_embeds)
        return SequenceBatch(
            higher_order_dom_embeds, dom_mask, left_justify=False), dom_elems

    @abc.abstractproperty
    def _query_embed_dim(self):
        """Returns the embedding dimension of each field / entry of the query
        (Int).
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _query_entries(self, states):
        """Returns the entries in the query (field values in structured
        query) otherwise words in natural language utterance.

        Args:
            state (list[MiniWoBState]): batch of states

        Returns:
            list[list[unicode]]: batch of entries
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _alignment_fields(self, states):
        """Returns the batch of alignment fields that will get used for DOM
        alignment.

        Args:
            states (list[MiniWoBState]): batch of states

        Returns:
            list[Fields]: batch of Fields that can be aligned to with
                DOM elements
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _type_values_and_probs(self, states, query_embeds, dom_contexts,
                               force_type_values=None):
        """Returns the possible values that type actions can take, along with
        their corresponding probs.

        Args:
            states (list[MiniWoBState]): batch of states
            query_embeds (SequenceBatch): embeds for each entry in
                the query shape = (batch, num_entries, embed_dim)
            dom_contexts (Variable[FloatTensor]): batch of context vectors
                shape = (batch, context_dim)
            force_type_values (list[unicode | None]): force these values to
                be scored if possible (optional)

        Returns:
            list[list[unicode]]: values to type NOT PADDED
            Variable[FloatTensor]: (batch, num_type_values) PADDED
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _query_embeds(self, states, query_entries):
        """Returns the query embeddings for a batch of states. This is either
        an embedding per word in the utterance or an embedding per field in
        the structrued query.

        Args:
            states (list[MiniWoBState]): batch of states
            query_entries (list[list[unicode]]): optimization, pass the
                entries of the query

        Returns:
            Variable[FloatTensor]: batch_size x num_words x embed_dim
        """
        raise NotImplementedError()

    # TODO: These could be factored out into a separate loss module?
    def update_from_replay_buffer(self, replay, gamma, take_grad_step):
        """
        
        Args:
            replay (ReplayBuffer)
            gamma (float)
            take_grad_step (Callable)

        Returns:
            loss (float)
            replay_trace (ReplayBufferUpdateTrace)
        """
        # TODO(ezliu): handle this properly
        try:
            episodes, replay_probs, sample_trace = replay.sample(
                    self._episodes_to_replay)
        except ReplayBufferNotReadyException:
            # don't do an update if the replay buffer is not ready
            # TODO: Return empty ReplayBufferTrace
            return None, None

        # we now score all episodes no matter what
        scored_episodes = self._score_episodes(episodes)

        # list of length batch_size
        if self._update_rule == "only-last-experience":
            model_prob = lambda episode: np.exp(episode[-1].log_prob.data.cpu().numpy()[0])
        elif self._update_rule == "use-whole-episode":
            model_prob = lambda episode: np.exp(np.sum(exp.log_prob.data.cpu().numpy()[0] for exp in episode))
        else:
            error_msg = "{} not a supported update rule".format(self._update_rule)
            raise NotImplementedError(error_msg)

        model_probs = [model_prob(ep) for ep in scored_episodes]

        # IMPORTANCE WEIGHTS ARE DISABLED: this yields better performance
        # importance_weights = [model_prob / replay_prob for model_prob,
        # replay_prob in zip(model_probs, replay_probs)]
        importance_weights = [1.] * len(model_probs)

        loss = self._loss_from_episodes(scored_episodes, gamma,
                                        importance_weights,
                                        use_critic=False)  # never use critic for replay buffer
        take_grad_step(loss)

        replay_trace = ReplayBufferUpdateTrace(scored_episodes, importance_weights, model_probs, gamma, sample_trace)
        return loss.data.sum(), replay_trace

    def update_from_episodes(self, episodes, gamma, take_grad_step):
        """Implements the REINFORCE loss, adjusted for sampling from the
        policy and takes a grad step.

            loss = - sum(gamma^t (G_t - baseline) log(p(z_t)))

        Args:
            episodes (list[Episode])
            gamma (float): the discount factor
            take_grad_step (Callable): takes a loss Variable and takes a
                gradient step on the loss
        """
        filtered_episodes = [ep for ep in episodes if not isinstance(
                ep[-1].action, MiniWoBTerminate)]
        if len(filtered_episodes) > 0:
            scored_episodes = self._score_episodes(filtered_episodes)
            imp_weights = [1.] * len(scored_episodes)
            loss = self._loss_from_episodes(scored_episodes, gamma, imp_weights,
                                            use_critic=self._use_critic)
            take_grad_step(loss)

    def update_from_demonstrations(self, demonstrations, take_grad_step):
        """Calculates the cross-entropy loss from a batch of demonstrations.

        Args:
            demonstrations (EpisodeGraph)

        Returns:
            loss (Variable[FloatTensor])
            take_grad_step (Callable): takes a loss Variable and takes a
                gradient step on the loss
        """
        experiences = flatten(episode_graph.to_experiences()
                for episode_graph in demonstrations)
        scored_experiences = self._score_experiences(experiences)
        loss = -sum(exp.log_prob for exp in scored_experiences) / len(demonstrations)
        self._clear_cache()
        take_grad_step(loss)

    def _loss_from_episodes(self, episodes, gamma, importance_weights, use_critic):
        """Actually implements loss.

        Args:
            episodes (list[Episode])
            gamma (float)
            importance_weights (list[float]): Of equal len to episodes. How
            much to reweight each episode by. (e.g. for importance sampling)
            use_critic (bool):
                - if False
                    - use empirical return in policy update
                    - do NOT update value function
                - if True
                    - use value function to estimate return in policy update
                    - update value function

        Returns:
            loss (Variable[FloatTensor])
        """
        assert len(episodes) == len(importance_weights)

        if use_critic:
            for w in importance_weights:
                assert w == 1.

        update_rule = self._update_rule

        total_loss = 0.
        for episode, imp_weight in zip(episodes, importance_weights):
            # this is not guaranteed to be right for episodes that prematurely terminate
            # in particular, computation of next_state_value and critic_loss
            assert not isinstance(episode[-1].action, MiniWoBTerminate)

            for t, exp in enumerate(episode):
                if update_rule == "only-last-experience":
                    if t + 1 != len(episode):
                        continue

                empirical_return = float(episode.discounted_return(t, gamma))

                # == ONE STEP RETURN ==
                # if t + 1 == len(episode):
                #     # implicit terminal state has value 0
                #     next_state_value = 0.
                # else:
                #     next_exp = episode[t + 1]
                #     next_state_value = next_exp.state_value
                # one_step_return = exp.undiscounted_reward + gamma * next_state_value

                if use_critic:
                    baseline = exp.state_value.detach()
                    # we do NOT want gradient going through this
                    # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/3
                else:
                    baseline = self._baseline

                advantage = empirical_return - baseline
                actor_loss = -exp.log_prob * advantage

                if use_critic:
                    # regress towards actual return, rather than 1-step return
                    critic_loss = (exp.state_value - empirical_return) ** 2
                else:
                    critic_loss = 0.

                total_loss = total_loss + (actor_loss + critic_loss) * imp_weight

        avg_loss = total_loss / len(episodes)
        self._clear_cache()

        return avg_loss

    def _build_attention(self, probs, elems):
        attention = np.zeros(
            (positions.OUTPUT_ROWS, positions.OUTPUT_COLS)).astype(np.float32)

        assert len(probs) == len(elems)
        for prob, elem in zip(probs, elems):
            if elem != DOMElementPAD():
                start_left, start_top = positions.grid_points(
                        elem.left, elem.top, "up", "up")
                end_left, end_top = positions.grid_points(
                        elem.left + elem.width, elem.top + elem.height,
                        "down", "down")
                attention[start_left: end_left + 1,
                          start_top: end_top + 1] += prob
        return attention

    def _score_episodes(self, episodes):
        """Score all the experiences in each of the episodes.

        Args:
            episodes (list[Episodes])

        Returns:
            scored_episodes (list[ScoredEpisode])
        """
        # score all experiences in a batch
        experiences = flatten(episodes)
        scored_experiences = self._score_experiences(experiences)

        # convert experiences back into episodes
        scored_episodes = []
        scored_experiences_reversed = list(reversed(scored_experiences))
        for ep in episodes:
            scored_ep = Episode()
            for _ in range(len(ep)):
                scored_ep.append(scored_experiences_reversed.pop())
            scored_episodes.append(scored_ep)

        return scored_episodes

    def _score_experiences(self, experiences):
        """Score experiences by the log prob of their actions.

        Args:
            experiences (list[Experience])

        Returns:
            list[ScoredExperience]
        """
        scored_experiences = []
        for exp_batch in as_batches(experiences, self._scoring_batch_size):
            states = [exp.state for exp in exp_batch]
            # force the second DOM attention to select the correct elements
            force_dom_attn = [exp.action.element for exp in exp_batch]
            force_type_values = [
                exp.action.text if isinstance(
                    exp.action, MiniWoBFocusAndType)
                else None for exp in exp_batch]
            action_scores_batch = self.score_actions(
                   states, force_dom_attn=force_dom_attn,
                   force_type_values=force_type_values)

            assert len(exp_batch) == len(action_scores_batch)
            for exp, action_scores in zip(exp_batch, action_scores_batch):
                log_prob = action_scores.as_variables[exp.action]
                state_value = action_scores.state_value
                scored_exp = ScoredExperience(exp.state, exp.action,
                                exp.undiscounted_reward, log_prob,
                                state_value, exp.metadata)
                scored_experiences.append(scored_exp)

        return scored_experiences

    def end_episodes(self):
        # TODO: There's a subtle issue here, where utterances will be padded
        # different lengths across different episodes. If you don't clear the
        # cache here, then you use the padding scheme from old episodes.
        # Theoretically, you could always pad to a max value, but this may be
        # hard to do.
        self._clear_cache()


class StructuredQueryPolicy(MiniWoBPolicy):
    """Implements MiniWoBPolicy, but embeds structured query fields instead
    of natural language utterance.
    """
    def __init__(self, episodes_to_replay, scoring_batch_size, update_rule,
                 baseline, use_critic, attn_dim, context_embed_dim,
                 dom_attention_for_state, utterance_embedder_config,
                 dom_embedder_config):
        super(StructuredQueryPolicy, self).__init__(
                episodes_to_replay, scoring_batch_size, update_rule,
                baseline, use_critic, attn_dim, context_embed_dim,
                dom_attention_for_state, utterance_embedder_config,
                dom_embedder_config)

        # Add the type value attention
        self._type_value_attention = Attention(
                self._query_embed_dim, context_embed_dim, attn_dim)

    @property
    def _query_embed_dim(self):
        return self._utterance_embedder.embed_dim * 2

    def _alignment_fields(self, states):
        """Aligns to all field (list[Fields])"""
        return [state.fields for state in states]

    def _type_values_and_probs(self, states, query_embeds, dom_contexts,
                               force_type_values=None):
        type_values = self._query_entries(states)
        type_value_probs = self._type_value_attention(
                query_embeds, dom_contexts).weights
        return type_values, type_value_probs

    def _query_entries(self, states):
        return [state.fields.values for state in states]

    def _query_embeds(self, states, query_entries):
        """Given a batch of states, embed the keys and values of each state's
        query.

        Args:
            states (list[MiniWoBState])

        Returns:
            entry_embeds (SequenceBatch): batch x num_keys x (2 * embed_dim)
                the keys and values concatenated
        """
        fields_batch = [state.fields for state in states]

        # list[list[list[unicode]]] (batch x num_keys x key length)
        values_batch = [[word_tokenize(value) for value in fields.values] for
                        fields in fields_batch]
        keys_batch = [[word_tokenize(key) for key in fields.keys] for fields
                      in fields_batch]

        # Pad
        batch_size = len(fields_batch)
        max_num_fields = max(len(values) for values in values_batch)
        max_num_fields = max(max_num_fields, 1)  # Ensure non-empty
        mask = torch.ones(batch_size, max_num_fields)
        assert len(keys_batch) == len(values_batch) == len(mask)
        for keys, values, submask in zip(keys_batch, values_batch, mask):
            assert len(keys) == len(values)
            if len(keys) < max_num_fields:
                submask[len(keys):] = 0.
                keys.extend(
                    [[UtteranceVocab.PAD] for _ in xrange(
                        max_num_fields - len(keys))])
                values.extend(
                    [[UtteranceVocab.PAD] for _ in xrange(
                        max_num_fields - len(values))])

        # Flatten to list[list[unicode]] (batch * num_keys) x key length
        keys_batch = flatten(keys_batch)
        values_batch = flatten(values_batch)

        # Embed and mask (batch * num_keys) x embed_dim
        key_embeds, _ = self._utterance_embedder(keys_batch)
        key_embeds = key_embeds.view(
                batch_size, max_num_fields, self._utterance_embedder.embed_dim)
        value_embeds, _ = self._utterance_embedder(values_batch)
        value_embeds = value_embeds.view(
                batch_size, max_num_fields, self._utterance_embedder.embed_dim)
        key_embeds = SequenceBatch(key_embeds, GPUVariable(mask))
        value_embeds = SequenceBatch(value_embeds, GPUVariable(mask))

        entry_embed_values = torch.cat(
                [key_embeds.values, value_embeds.values], 2)
        entry_embeds = SequenceBatch(entry_embed_values, key_embeds.mask)
        return entry_embeds


class NaturalLanguagePolicy(MiniWoBPolicy):
    """Implements the MiniWoBPolicy but embeds the natural language instead
    of structured query.
    """
    def __init__(self, episodes_to_replay, scoring_batch_size, update_rule,
                 baseline, use_critic, attn_dim, context_embed_dim,
                 dom_attention_for_state, utterance_embedder_config,
                 dom_embedder_config):
        super(NaturalLanguagePolicy, self).__init__(
                episodes_to_replay, scoring_batch_size, update_rule,
                baseline, use_critic, attn_dim, context_embed_dim,
                dom_attention_for_state, utterance_embedder_config,
                dom_embedder_config)

        # Predict the start and end index separately
        self._type_start = Attention(
                self._query_embed_dim, context_embed_dim, attn_dim)
        self._type_end = Attention(
                self._query_embed_dim, context_embed_dim, attn_dim)

        # Choose the top k start and end indices
        self._k = 3

    @property
    def _query_embed_dim(self):
        return self._utterance_embedder.embed_dim

    def _alignment_fields(self, states):
        """Returns state utterances wrapped in a Fields (list[Fields])"""
        return [Fields({"utterance": state.utterance}) for state in states]

    def _type_values_and_probs(self, states, query_embeds, dom_contexts,
                               force_type_values=None):
        # (batch, num_words)
        start_index_probs_batch = self._type_start(
                query_embeds, dom_contexts).weights

        # (batch, num_words)
        end_index_probs_batch = self._type_end(
                query_embeds, dom_contexts).weights

        if force_type_values:
            # Must be a full batch of forced type values
            type_values_batch = []
            start_indices = []  # inclusive
            end_indices = []    # inclusive
            assert len(force_type_values) == len(states)
            for batch_idx, (value, state) in enumerate(
                    zip(force_type_values, states)):
                if value is None:
                    type_values_batch.append([])
                    # Choose arbitrary indices
                    start_indices.append(0)
                    end_indices.append(0)
                else:
                    type_values_batch.append([value])
                    # Must be a valid span
                    tokenized_value = word_tokenize(value)
                    tokenized_utterance = word_tokenize(state.utterance)
                    start_index = utils.find_sublist(
                            tokenized_utterance, tokenized_value)
                    assert start_index != -1
                    start_indices.append(start_index)
                    end_indices.append(start_index + len(tokenized_value) - 1)

            # Convert to tensors (batch, 1)
            start_indices = GPUVariable(torch.unsqueeze(
                    torch.LongTensor(start_indices), 1))
            end_indices = GPUVariable(
                    torch.unsqueeze(torch.LongTensor(end_indices), 1))

            start_probs = torch.gather(
                    start_index_probs_batch, 1, start_indices)
            end_probs = torch.gather(
                    end_index_probs_batch, 1, end_indices)

            # (batch, 1)
            type_value_probs_batch = start_probs * end_probs
        else:
            # (batch, k)
            batch_size, num_words = start_index_probs_batch.size()
            k = min(self._k, num_words)
            top_k_start_probs, top_k_start_indices = torch.topk(
                    start_index_probs_batch, k, 1)
            top_k_end_probs, top_k_end_indices = torch.topk(
                    end_index_probs_batch, k, 1)

            # reshape the probs for outer product via bmm
            # (batch, k, 1)
            top_k_start_probs = torch.unsqueeze(top_k_start_probs, 2)
            # (batch, 1, k)
            top_k_end_probs = torch.unsqueeze(top_k_end_probs, 1)
            type_value_probs_batch = torch.bmm(
                    top_k_start_probs, top_k_end_probs)
            # flatten to (batch, k^2)
            type_value_probs_batch = type_value_probs_batch.view(
                    batch_size, k * k)

            # Calculate the corresponding type actions
            type_values_batch = []
            mask = np.zeros((batch_size, k * k)).astype(np.float32)
            for batch_idx in xrange(batch_size):
                state = states[batch_idx]
                type_values = []
                for i, start_index in enumerate(top_k_start_indices[batch_idx].data.cpu().numpy()):
                    for j, end_index in enumerate(top_k_end_indices[batch_idx].data.cpu().numpy()):
                        if start_index <= end_index < len(state.tokens):
                            type_values.append(
                                    state.detokenize(start_index, end_index + 1))
                            mask[batch_idx, i * k + j] = 1.
                        else:
                            type_values.append("INVALID_TYPE_ACTION")
                type_values_batch.append(type_values)

            # Mask out the entries where start > end
            mask = GPUVariable(torch.from_numpy(mask))
            type_value_probs_batch = type_value_probs_batch * mask

            # Renormalize l1 norm over first dim to 1
            type_value_probs_batch = torch.renorm(
                    type_value_probs_batch, 1, 1, 1)
        return type_values_batch, type_value_probs_batch

    def _query_entries(self, states):
        return [state.tokens for state in states]

    def _query_embeds(self, states, query_entries):
        _, query_elems = self._utterance_embedder(query_entries)
        values = torch.stack([e.values for e in query_elems], 0)
        mask = torch.stack([torch.squeeze(e.mask, 1) for e in query_elems], 0)
        query_embeds = SequenceBatch(values, mask)
        return query_embeds


class Selection(namedtuple('Selection', ['selected', 'probs', 'indices', 'candidates', 'candidate_probs'])):
    """
    Attributes:
        selected (list[Object]): a batch of items selected (some items can be None)
        probs (Variable[FloatTensor]): of shape (batch_size). The selection
            probabilities corresponding to each item.
        indices (Variable[LongTensor]): of shape (batch_size). The index of
            each corresponding item. Indexing is per-batch-item, not per-batch.
    """
    def __new__(cls, candidate_selector, candidate_probs, candidates=None):
        """Select candidates.

        Args:
            candidate_selector (Callable[[Variable[FloatTensor]], list[int]]):
                takes candidate_probs and returns a batch of selections (integers)
            candidate_probs (Variable[FloatTensor]): of shape (batch_size, num_candidates)
            candidates (list[list[object]]): a batch of candidate sets, where each
                set is a list of candidates
        """
        indices = candidate_selector(candidate_probs)
        if candidates is not None:
            assert len(candidates) == len(indices)
            selected = [thing_list[index] for thing_list, index in zip(
                candidates, indices)]
        else:
            selected = None

        indices = GPUVariable(torch.LongTensor(indices))  # (batch_size,)

        probs = torch.gather(candidate_probs, 1, torch.unsqueeze(indices, 1))  # (batch_size, 1)
        probs = torch.squeeze(probs, 1)  # (batch_size)

        cls._check_shapes(selected, probs, indices)

        self = super(Selection, cls).__new__(cls, selected, probs, indices, candidates, candidate_probs)
        return self

    @classmethod
    def _check_shapes(cls, selected, probs, indices):
        assert probs.size() == indices.size()
        assert len(probs.size()) == 1
        batch_size = probs.size()[0]
        if selected:
            assert len(selected) == batch_size


class MiniWoBPolicyJustification(Justification):
    """Justifications generated by MiniWoBPolicy."""

    def __init__(self, dom_elements, element_probs,
                 click_or_type_probs, query_entries, fields_attentions,
                 type_values, type_value_probs, state_value):
        """
        Args:
            dom_elements (list[DOMElement]): may contain DOMElementPAD
            element_probs (Variable[FloatTensor]): of shape (num_elems,)
            click_or_type_probs (Variable[FloatTensor]): of shape (2,)
            query_entries (list[unicode]): entries in the query, structured
                or otherwise
            fields_attentions (list[Variable[FloatTensor]): attention over
                values from DOMElement, of shape (num_values,) of length 2
                (two heads)
            type_values (list[unicode]): possible values to type
            type_value_probs (Variable[FloatTensor]): of shape (num_values,)
            state_value (Variable): scalar
        """
        self._dom_elements = dom_elements
        self._element_probs = to_numpy(element_probs)

        self._click_or_type_probs = to_numpy(click_or_type_probs)
        self._query_entries = query_entries
        self._fields_attentions = [to_numpy(fa) for fa in fields_attentions]
        self._type_values = type_values
        self._type_value_probs = to_numpy(type_value_probs)
        assert len(type_value_probs.size()) == 1
        self._state_value = float(to_numpy(state_value))

    def to_json_dict(self):
        def as_pairs(items, probs):
            item_strs = [unicode(item) for item in items]
            prob_floats = [round(f, 4) for f in probs]
            return sorted(zip(item_strs, prob_floats), key=lambda x: -x[1])

        return {
            'elements': as_pairs(self.dom_elements, self.element_probs),
            'value attentions 1': as_pairs(
                self._query_entries, self._fields_attentions[0]),
            'value attentions 2': as_pairs(
                self._query_entries, self._fields_attentions[1]),
            'action type': as_pairs(
                ['CLICK', 'TYPE'], self._click_or_type_probs),
            'typed': as_pairs(self._type_values, self._type_value_probs),
            'state value': self._state_value,
        }

    @property
    def dom_elements(self):
        return self._dom_elements

    @property
    def element_probs(self):
        return self._element_probs

    def dumps(self):
        values_attn_str = self._pretty_string(
                self._query_entries, self._fields_attentions[0])
        values_attn_str2 = self._pretty_string(
                self._query_entries, self._fields_attentions[1])
        elems_str = self._pretty_string(self.dom_elements, self.element_probs)
        type_str = self._pretty_string(
                ["CLICK", "TYPE"], self._click_or_type_probs)
        values_str = self._pretty_string(
                self._type_values, self._type_value_probs)

        return (u"elements:\n{}\nvalue attentions:\n{}\n"
                u"value attentions 2:\n{}\naction type:\n{}\n"
                u"typed:\n{}\nstate value: {:.3f}").format(
                    indent(elems_str), indent(values_attn_str),
                    indent(values_attn_str2), indent(type_str),
                    indent(values_str), self._state_value)

    @classmethod
    def _pretty_string(cls, things, thing_probs, selected_thing=None):
        """Given a list of objects with their corresponding probabilities and
        the selected one, returns a pretty printed string.

        Args:
            things (list[Object])
            thing_probs (list[float] | Variable)
            selected_thing (Object)

        Returns:
            string
        """
        items = []
        for thing, prob in sorted(
                zip(things, thing_probs),
                key=lambda (thing, prob): prob, reverse=True):
            selected = '>' if thing == selected_thing else ' '
            items.append(u'{} [{:.3f}] {}'.format(selected, prob, thing))
        return '\n'.join(items)


class ReplayBufferUpdateTrace(Trace):
    def __init__(self, scored_episodes, importance_weights, model_probs, discount_factor, sample_trace):
        """ReplayBufferUpdateTrace.

        Args:
            scored_episodes (list[Episode])
            importance_weights (list[float]): of same length as scored_episodes
            model_probs (list[float]): of same length as scored_episodes. Probability of the episode under the model.
            discount_factor (float)
            sample_trace (ReplayBufferSampleTrace)
        """
        assert len(scored_episodes) == len(importance_weights)
        self._scored_episodes = scored_episodes
        self._importance_weights = importance_weights
        self._model_probs = model_probs
        self._discount_factor = discount_factor
        self._sample_trace = sample_trace

    def to_json_dict(self):
        return {
            'episodes': [MiniWoBEpisodeTrace(ep).to_json_dict() for ep in self._scored_episodes],
            'importance_weights': self._importance_weights,
            'model_probs': self._model_probs,
            'discount_factor': self._discount_factor,
            'sampling': self._sample_trace.to_json_dict()
        }

    def dumps(self):
        weighted_episodes = zip(self._scored_episodes, self._importance_weights, self._model_probs)
        weighted_episodes.sort(key=lambda x: -x[1])  # sort by importance weight

        episodes_str = u'\n\n'.join(
            u'===== EPISODE {i} (importance weight = {importance:.3f}, model prob = {model:.3f}) =====\n\n{episode}'
            .format(i=i, importance=importance, model=model_prob, episode=indent(MiniWoBEpisodeTrace(ep).dumps()))
            for i, (ep, importance, model_prob) in enumerate(weighted_episodes)
        )
        return u'Replay buffer:\n{samples}\n\nImportance weights:\n{weights}\n\n{episodes}'.format(
            samples=indent(self._sample_trace.dumps() if self._sample_trace else None),
            weights=sorted([round(w, 3) for w in self._importance_weights], reverse=True),
            episodes=episodes_str,
        )

