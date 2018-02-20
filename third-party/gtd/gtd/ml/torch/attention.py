from collections import namedtuple

import math
import torch
import numpy as np
from gtd.ml.torch.utils import GPUVariable
from torch.nn import Parameter
from torch.nn import Softmax, Tanh, Module

from gtd.ml.torch.seq_batch import SequenceBatch
from gtd.ml.torch.utils import conditional, NamedTupleLike


class AttentionOutput(namedtuple('AttentionOutput', ['weights', 'context', 'logits']), NamedTupleLike):
    pass
"""
Attributes:
    weights (Variable): of shape (batch_size, num_cells)
    context (Variable): of shape (batch_size, memory_dim)
    logits (Variable): of shape (batch_size, num_cells), the weights before
        they have been softmaxed
"""

class DummyAttention(Module):
    def __init__(self, memory_dim, query_dim, attn_dim):
        super(DummyAttention, self).__init__()
        self.memory_dim = memory_dim
        self.query_dim = query_dim
        self.attn_dim = attn_dim

    def forward(self, memory_cells, query):
        batch_size, num_cells = memory_cells.mask.size()
        logits = GPUVariable(torch.zeros(batch_size, num_cells))
        weights = GPUVariable(torch.zeros(batch_size, num_cells))
        context = GPUVariable(torch.zeros(batch_size, self.memory_dim))
        return AttentionOutput(weights=weights, context=context, logits=logits)

class Attention(Module):
    def __init__(self, memory_dim, query_dim, attn_dim):
        super(Attention, self).__init__()
        self.tanh = Tanh()
        self.softmax = Softmax()

        self.memory_dim = memory_dim
        self.query_dim = query_dim
        self.attn_dim = attn_dim

        self.memory_transform = Parameter(self._initialize_weight_matrix(memory_dim, attn_dim))  # Wh
        self.query_transform = Parameter(self._initialize_weight_matrix(query_dim, attn_dim))  # Ws
        self.v_transform = Parameter(self._initialize_weight_matrix(attn_dim, 1))  # v

    @classmethod
    def _initialize_weight_matrix(cls, in_dim, out_dim):
        stdv = 1. / math.sqrt(in_dim)
        m = torch.ones(in_dim, out_dim)
        m.uniform_(-stdv, stdv)
        return m

    def forward(self, memory_cells, query):
        """Generates a density over a set of elements w.r.t. the query vector.

        Et(i) = tanh(Hi * Wh + St * Ws) * v
        At = softmax(Et)

        Dimensions:
            Hi: (batch_size x memory_dim)
            St: (batch_size x query_dim)
            Wh: (memory_dim x attn_dim)
            Ws: (query_dim x attn_dim)
            v:  (attn_dim x 1)
            --
            tanh( Hi * Wh + St * Ws ):       (batch_size x attn_dim)
            tanh( Hi * Wh + St * Ws ) * v:   (batch_size x 1)
            At = softmax(Et):                (batch_size x num_cells)

        Args:
            memory_cells (SequenceBatch): (batch_size x num_cells x memory_dim)
            query (torch.Variable): St (batch_size x query_dim)

        Returns:
            Variable: (batch_size x num_cells) array
        """
        transformed_query = torch.mm(query, self.query_transform)  # (batch_size, attn_dim)

        batch_size, num_cells = memory_cells.mask.size()
        memory_cells_ = torch.transpose(memory_cells.values, 0, 1)  # (num_cells, batch_size, memory_dim)
        expanded_transformed_query = transformed_query.expand(num_cells, batch_size, self.attn_dim)
        expanded_memory_transform = self.memory_transform.expand(num_cells, self.memory_dim, self.attn_dim)
        expanded_v_transform = self.v_transform.expand(num_cells, self.attn_dim, 1)

        # (num_cells, batch_size, attn_dim)
        attn_embeds = torch.bmm(memory_cells_, expanded_memory_transform) + expanded_transformed_query
        attn_embeds = self.tanh(attn_embeds)
        attn_embeds = torch.bmm(attn_embeds, expanded_v_transform)  # (num_cells, batch_size, 1)
        logits = torch.transpose(attn_embeds.squeeze(2), 0, 1)
        logits = self._mask_logits(logits, memory_cells.mask)

        # compute normalized weights
        weights = self.softmax(logits)  # (batch_size, num_cells)
        weights = self._mask_weights(weights, memory_cells.mask)

        context = self._context_from_weights(weights, memory_cells)
        return AttentionOutput(weights=weights, context=context, logits=logits)

    @classmethod
    def _no_cells(cls, mask):
        # no_cells is a FloatTensor with shape (batch_size, num_cells)
        # no_cells[i, j] = 1 if example i has NO memory cells, 0 otherwise

        # TODO(kelvin): check for numerical stability. Product of 1's does
        #   not necessarily equal 1 exactly, which we need

        no_cells = (1 - mask).prod(1).expand_as(mask)
        return no_cells

    @classmethod
    def _mask_logits(cls, logits, mask):
        no_cells = cls._no_cells(mask)  # (batch_size, num_cells)
        suppress = GPUVariable(torch.zeros(*mask.size()))

        # send the logit of non-cells to -infinity
        suppress[mask == 0] = float('-inf')
        # but if an entire row has no cells, just leave the cells alone
        suppress[no_cells == 1] = 0.0

        logits = logits + suppress  # -inf + anything = -inf

        return logits

    @classmethod
    def _mask_weights(cls, weights, mask):
        # if a given row has no memory cells, weights should be all zeros
        no_cells = cls._no_cells(mask)
        all_zeros = GPUVariable(torch.zeros(*mask.size()))
        weights = conditional(no_cells, all_zeros, weights)
        return weights

    @classmethod
    def _context_from_weights(cls, weights, memory_cells):
        context = torch.bmm(weights.unsqueeze(1), memory_cells.values)  # (batch_size, 1, memory_dim)
        context = context.squeeze(1)  # (batch_size, memory_dim)
        return context


class SentinelAttention(Attention):
    """Performs sentinel attention: adds the sentinel vector into the memory
    cells so that the attention weights over the original memory cells do not
    necessarily sum to 1.
    """
    def __init__(self, memory_dim, query_dim, attn_dim, sentinel_embed):
        """
        Args:
            memory_dim (int): dim of memory cells
            query_dim (int): dim of query
            attn_dim (int): dim of attn vector
            sentinel_embed (Variable[FloatTensor]): (memory_dim,) vector for
                sentinel. Should generally pass with requires_grad=False
        """
        super(SentinelAttention, self).__init__(
                memory_dim, query_dim, attn_dim)

        sentinel_size = sentinel_embed.size()
        assert len(sentinel_size) == 1
        assert sentinel_size[0] == memory_dim
        self._sentinel_embed = sentinel_embed

    def forward(self, memory_cells, query):
        """Performs sentinel attention with a sentinel of 0. Returns the
        AttentionOutput where the weights do not include the sentinel weight.

        Args:
            memory_cells (Variable[FloatTensor]): batch x num_cells x cell_dim
            query (Variable[FloatTensor]): batch x query_dim

        Returns:
            AttentionOutput: weights do not include sentinel weights
        """
        batch_size, _, cell_dim = memory_cells.values.size()
        sentinel = self._sentinel_embed.expand(
                batch_size, 1, cell_dim)
        sentinel_mask = GPUVariable(torch.ones(batch_size, 1))

        cell_values_with_sentinel = torch.cat(
            [memory_cells.values, sentinel], 1)
        cell_masks_with_sentinel = torch.cat(
            [memory_cells.mask, sentinel_mask], 1)
        cells_with_sentinel = SequenceBatch(
            cell_values_with_sentinel, cell_masks_with_sentinel,
            left_justify=False)

        attention_output = super(SentinelAttention, self).forward(
                cells_with_sentinel, query)
        weights_with_sentinel = attention_output.weights

        # TODO: Bring this line in after torch v0.2.0
        # weights_without_sentinel = weights_with_sentinel[batch_size, :-1]
        # attention_output = AttentionOutput(
        #   weights=weights_without_sentinel, context=attention_output.context)
        return attention_output


class SoftCopyAttentionOutput(namedtuple('SoftCopyAttentionOutput',
    ['weights', 'context', 'logits', 'orig_logits', 'boost']), NamedTupleLike):
    pass
    """
    Attributes:
        weights (Variable): of shape (batch_size, num_cells)
        context (Variable): of shape (batch_size, memory_dim)
        logits (Variable): of shape (batch_size, num_cells), the weights before
            they have been softmaxed
        orig_logits (Variable): of shape (batch_size, num_cells), the logits
            before they were boosted
        boost (Variable): the amount by which we boost the exp logits
    """


class SoftCopyAttention(Module):
    def __init__(self, memory_dim, query_dim, attn_dim):
        super(SoftCopyAttention, self).__init__()
        self._base_attention = Attention(memory_dim, query_dim, attn_dim)

    @staticmethod
    def _is_subset(a, b):
        """Check that boolean tensor a is a subset of b."""
        return torch.prod(b - a >= 0).data.sum() == 1

    def forward(self, memory_cells, query, alignments, copy_source):
        """Compute attention with soft-copy.

        Args:
            memory_cells (SequenceBatch): of shape (batch_size, num_cells, memory_dim)
            query (Variable): of shape (batch_size, query_dim)
            alignments (SequenceBatch): int-valued, of shape
                (batch_size, num_cells). If something has no alignment, it will
                have value 0 in the mask.
            copy_source (Variable): of shape (batch_size, num_candidates)

        This behaves like normal attention, except we boost the
        exponentiated logits:

        exp_logits[i][j] += copy_source[i][alignments[i][j]]

        This is inspired by:
            "Incorporating Copying Mechanism in Sequence-to-Sequence Learning"
            http://www.aclweb.org/anthology/P16-1154

        Returns:
            AttentionOutput
        """
        if not self._is_subset(alignments.mask, memory_cells.mask):
            raise ValueError('Alignments mask must be a subset of memory cells mask.')

        base_attn = self._base_attention(memory_cells, query)  # AttentionOutput

        exp_logits = torch.exp(base_attn.logits)  # (batch_size, num_cells)

        # y = torch.gather(x, dim=1, index=index)
        # y[i][j] = x[i][index[i][j]]
        boost = torch.gather(copy_source, dim=1, index=alignments.values)

        # no boost for items with no alignment
        boost = boost * alignments.mask

        # boost the exponentiated logits
        boosted_exp_logits = exp_logits + boost

        # normalize to compute final weights
        normalizer = torch.sum(boosted_exp_logits, 1).expand_as(boosted_exp_logits)
        # (batch_size, num_cells)

        weights = boosted_exp_logits / normalizer
        weights = Attention._mask_weights(weights, memory_cells.mask)

        if not np.isfinite(weights.data.sum()):
            raise ValueError('Some attention weights are NaN')
            # TODO(kelvin): need to avoid numerical precision issues
            # TODO(kelvin): need to avoid division by zero

        # compute context
        context = Attention._context_from_weights(weights, memory_cells)

        logits = torch.log(boosted_exp_logits)
        return SoftCopyAttentionOutput(
            weights=weights, context=context, logits=logits,
            orig_logits=base_attn.logits, boost=boost
        )


class SoftCopyAttentionTrace(object):
    def __init__(self, soft_copy_attn_out, idx,
                 memory_elements, copy_elements, alignments):
        """Visualize soft copy attention.
        
        Args:
            soft_copy_attn_out (SoftCopyAttentionOutput)
            idx (int): batch index
            memory_elements (list[object])
            copy_elements (list[object])
            alignments (SequenceBatch)
        """
        self._weights = soft_copy_attn_out.weights.data[idx]
        self._memory_elements = memory_elements

        self._orig_logits = soft_copy_attn_out.orig_logits.data[idx]  # (num_cells,)

        self._boost = soft_copy_attn_out.boost.data[idx]  # (num_cells,)

        self._boost_elements = []
        alignment_vals = alignments.values.data[idx]
        alignment_mask = alignments.mask.data[idx]
        for j, mask_val in zip(alignment_vals, alignment_mask):
            elem = copy_elements[j] if mask_val == 1.0 else None
            self._boost_elements.append(elem)

    def __repr__(self):
        format_str = u'{weight:.2f} = [exp({orig_logit:.2f}) + {boost:.2f}]/Z {elem} <-> {boost_elem}'

        combined = zip(self._memory_elements, self._weights, self._orig_logits,
                       self._boost, self._boost_elements)

        # sort by descending weight
        combined = sorted(combined, key=lambda x: x[1], reverse=True)

        lines = []
        for elem, weight, orig_logit, boost, boost_elem in combined:
            line = format_str.format(
                weight=weight, orig_logit=orig_logit, boost=boost,
                elem=elem, boost_elem=boost_elem)
            lines.append(line)

        return u'\n'.join(lines)

    def to_json_dict(self):
        return {}  # TODO(kelvin): implement