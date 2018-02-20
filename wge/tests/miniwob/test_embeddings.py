import pytest
import torch
from torch.nn import Module
import numpy as np

from gtd.ml.torch.utils import GPUVariable, assert_tensor_equal
from wge.miniwob.embeddings import HigherOrderDOMElementEmbedder, DOMElementPAD


class DummyDOMElement(object):
    def __init__(self, value):
        self.value = value
        self.parent = None
        self.children = []

    def __repr__(self):
        return self.value


class DummyBaseDOMEmbedder(Module):
    def forward(self, dom_elems):
        # hard-coded. Actually just ignores dom_elems

        return GPUVariable(torch.FloatTensor([
            [[0.1, 0], [1, 1], [2, 2], [3, 3], [1, 0], [1, 0.5], [2, 0]],
            [[1, 1], [2, 2], [3, 3], [1, 0], [0, 0], [0, 0], [0, 0]],
            [[1, 0], [2, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        ]))


class TestHigherOrderDOMElementEmbedder(object):
    @pytest.fixture
    def dom_elements(self):
        def set_children(parent, children):
            parent.children = list(children)
            for child in children:
                child.parent = parent

        d0 = DummyDOMElement('d0')  # no siblings
        d1 = DummyDOMElement('d1')  # two siblings
        d2 = DummyDOMElement('d2')
        d3 = DummyDOMElement('d3')
        set_children(d0, [d1, d2, d3])

        d1a = DummyDOMElement('d1a')  # one sibling
        d1b = DummyDOMElement('d1b')
        set_children(d1, [d1a, d1b])

        d2a = DummyDOMElement('d2a')  # sibling exists, but not included in returned list
        d2b = DummyDOMElement('d2b')  # NOT included in returned list
        set_children(d2, [d2a, d2b])

        return [d0, d1, d2, d3, d1a, d1b, d2a]

    @pytest.fixture
    def embedder(self):
        return HigherOrderDOMElementEmbedder(DummyBaseDOMEmbedder())

    def test_forward(self, embedder, dom_elements):
        d0, d1, d2, d3, d1a, d1b, d2a = dom_elements
        pad = DOMElementPAD()

        # a batch of DOM element sequences
        # one unrealistic aspect of this batch is that different sequences share the same elements
        # (won't happen in the actual task)
        batch = [
            [d0, d1, d2, d3, d1a, d1b, d2a],
            [d1, d2, d3, d1a, pad, pad, pad],
            [d1a, d2a, pad, pad, pad, pad, pad],
        ]

        # the correct embeddings for each DOM element:
        # the original embedding, concatenated with the average of your neighbors

        d0_ = [0.1, 0, 0, 0]  # d0 should have no neighbors
        d1_ = [1, 1, 2.5, 2.5]
        d2_ = [2, 2, 2, 2]
        d3_ = [3, 3, 1.5, 1.5]
        d1a_ = [1, 0, 1, 0.5]
        d1b_ = [1, 0.5, 1, 0]
        d2a_ = [2, 0, 0, 0]  # d2a should have no neighbors
        pad_ = [0, 0, 0, 0]

        correct = np.array([
            [d0_,  d1_,  d2_,   d3_,   d1a_,  d1b_,  d2a_],
            [d1_,  d2_,  d3_,   d1a_,  pad_, pad_, pad_],
            [d1a_, d2a_, pad_, pad_, pad_, pad_, pad_],
        ], dtype=np.float32)

        result = embedder(batch)
        assert_tensor_equal(result, correct)
