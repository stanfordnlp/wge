import numpy as np
import pytest
import torch

from gtd.ml.torch.attention import Attention, SoftCopyAttention, AttentionOutput
from gtd.ml.torch.seq_batch import SequenceBatch
from gtd.ml.torch.utils import GPUVariable
from gtd.ml.torch.utils import assert_tensor_equal
from gtd.utils import cached_property

float_tensor = lambda arr: torch.FloatTensor(arr)
float_tensor_var = lambda arr: GPUVariable(float_tensor(arr))


class AttentionExample(object):
    @cached_property
    def params(self):
        memory_transform = np.array([  # Wh: (memory_dim x attn_dim)
            [.1, .5],
            [.2, .6],
            [.3, .7],
            [.4, .8],
        ])
        query_transform = np.array([  # Ws: (query_dim x attn_dim)
            [.3, .4],
            [.2, .5],
            [.1, .6],
        ])
        v_transform = np.array([  # v: (attn_dim x 1)
            [.1],
            [.8],
        ])
        return memory_transform, query_transform, v_transform

    @cached_property
    def memory_cells(self):
        mem_values = float_tensor_var([  # (batch_size x num_cells x memory_dim)
            [
                [.1, .2, .3, .4],
                [.4, .5, .6, .7],
            ],
            [
                [.2, .3, .4, .5],
                [.6, .7, .8, .9],
            ],
            [
                [.3, .4, .5, .6],
                [.7, .8, .9, .1],
            ],
            [
                [-8, -9, -10, -11],
                [-12, -13, -14, -15],
            ],
            [
                [8, 9, 10, 11],
                [12, 13, 14, 15],
            ]
        ])
        mem_mask = float_tensor_var([
            [1, 0],
            [1, 1],
            [1, 0],
            [0, 0],  # empty row
            [0, 1],  # right-justified
        ])
        memory_cells = SequenceBatch(values=mem_values, mask=mem_mask,
                                     left_justify=False)
        return memory_cells

    @cached_property
    def query(self):
        query = float_tensor_var([  # (batch_size x query_dim)
            [.1, .2, .3],
            [.4, .5, .6],
            [.7, .8, .9],
            [10, 11, 12],
            [13, 14, 15],
        ])
        return query

    @cached_property
    def correct_logits(self):
        # manual_logits = np.array([[0.65388812, 0.81788159],
        #                           [0.81039669, 0.87306204],
        #                           [0.86236411, 0.86977563]])

        ninf = -float('inf')
        manual_logits = float_tensor_var([[0.65388812, ninf],
                                          [0.81039669, 0.87306204],
                                          [0.86236411, ninf],
                                          [-0.899851, -0.9],
                                          [ninf, 0.9]
                                          ])

        return manual_logits

    @cached_property
    def correct_weights(self):
        # compute manually
        manual_weights = float_tensor_var([[1., 0.],
                                           [0.4843, 0.5156],
                                           [1., 0.],
                                           [0, 0],  # zero probability
                                           [0, 1.],
                                           ])
        return manual_weights

    @cached_property
    def correct_context(self):
        manual_context = float_tensor_var([[0.1, 0.2, 0.3, 0.4],
                                           [0.4062, 0.5062, 0.6062, 0.7062],
                                           [0.3, 0.4, 0.5, 0.6],
                                           [0, 0, 0, 0],  # zero vector
                                           [12, 13, 14, 15],
                                           ])
        return manual_context


class TestAttention(object):
    def test_forward(self):
        memory_dim, query_dim, attn_dim = 4, 3, 2
        attn = Attention(memory_dim, query_dim, attn_dim)

        attn_ex = AttentionExample()
        memory_transform, query_transform, v_transform = attn_ex.params
        memory_cells = attn_ex.memory_cells
        query = attn_ex.query

        # manually set Module parameters
        attn.memory_transform.data.set_(float_tensor(memory_transform))
        attn.query_transform.data.set_(float_tensor(query_transform))
        attn.v_transform.data.set_(float_tensor(v_transform))

        # compute with module
        attn_out = attn(memory_cells, query)
        assert_tensor_equal(attn_out.weights, attn_ex.correct_weights,
                            decimal=4)
        assert_tensor_equal(attn_out.context, attn_ex.correct_context,
                            decimal=4)
        assert_tensor_equal(attn_out.logits, attn_ex.correct_logits, decimal=4)


class TestSoftCopyAttention(object):
    @pytest.fixture
    def copy_source(self):
        return float_tensor_var([
            [.0, .2, .4, .6],
            [.1, .3, .5, .7],
            [.1, .2, .3, .4],
            [.01, .02, .03, .04],
            [.01, .03, .05, .07],
        ])

    @pytest.fixture
    def alignments(self):
        values = GPUVariable(torch.LongTensor([
            [1, 3],
            [1, 1],
            [3, 2],
            [3, 0],
            [0, 0],
        ]))

        mask = float_tensor_var([
            [1, 0],
            [1, 1],
            [0, 0],  # subset of memory cell mask
            [0, 0],
            [0, 1],
        ])

        return SequenceBatch(values, mask, left_justify=False)

    def test_is_subset(self):
        a = float_tensor_var([
            [1, 1, 0],
            [0, 1, 0],
        ])

        b = float_tensor_var([
            [1, 1, 0],
            [0, 1, 1],
        ])

        c = float_tensor_var([
            [1, 1, 0],
            [0, 0, 1],
        ])

        assert SoftCopyAttention._is_subset(a, a)  # you are a subset of self

        assert SoftCopyAttention._is_subset(a, b)  # b contains a
        assert not SoftCopyAttention._is_subset(b, a)  # a does not contain b

        assert not SoftCopyAttention._is_subset(a, c)  # a does not contain c
        assert not SoftCopyAttention._is_subset(c, a)  # c does not contain a

    def test_forward(self, copy_source, alignments):
        memory_dim, query_dim, attn_dim = 4, 3, 2
        attn = SoftCopyAttention(memory_dim, query_dim, attn_dim)

        attn_ex = AttentionExample()
        memory_transform, query_transform, v_transform = attn_ex.params
        memory_cells = attn_ex.memory_cells
        query = attn_ex.query

        # manually set parameters of the base attention
        base_attn = attn._base_attention
        base_attn.memory_transform.data.set_(float_tensor(memory_transform))
        base_attn.query_transform.data.set_(float_tensor(query_transform))
        base_attn.v_transform.data.set_(float_tensor(v_transform))

        # compute correct logits
        exp_logits = torch.exp(attn_ex.correct_logits)
        boost = float_tensor_var([
            [.2, 0],
            [.3, .3],
            [0, 0],
            [0, 0],
            [0, .01],
        ])
        correct_logits = torch.log(exp_logits + boost)

        # compute with module
        attn_out = attn(memory_cells, query, alignments, copy_source)

        assert_tensor_equal(attn_out.logits, correct_logits)
        assert_tensor_equal(attn_out.orig_logits, attn_ex.correct_logits)
        assert_tensor_equal(attn_out.boost, boost)