import numpy as np
import torch

from gtd.ml.torch.utils import assert_tensor_equal, GPUVariable
from gtd.utils import Bunch
from wge.wob_policy import MiniWoBPolicy


class TestMiniWoBPolicy(object):
    def test_states_to_image_var(self):
        # mock a MiniWoBState object
        arr_to_state = lambda arr: Bunch(observation=Bunch(image=np.array(arr, dtype=np.float32)))
        arrs = [
            # ex0
            [
                [[1, 2],  # R
                 [3, 4]],
                [[1, 0],  # G
                 [3, 0]],
                [[0, 2],  # B
                 [0, 4]],
            ],
            # ex1
            [
                [[10, 20],  # R
                 [30, 40]],
                [[10, 0],  # G
                 [30, 0]],
                [[0, 20],  # B
                 [0, 40]],
            ],
            # ex2
            [
                [[100, 200],  # R
                 [300, 400]],
                [[100, 0],  # G
                 [300, 0]],
                [[0, 200],  # B
                 [0, 400]],
            ]
        ]

        correct = np.array(arrs, dtype=np.float32)

        states = [arr_to_state(arr) for arr in arrs]
        image_var = MiniWoBPolicy._states_to_image_var(states)
        assert_tensor_equal(correct, image_var)

    def test_sample_elements(self):
        # sampling should be deterministic in this case
        element_probs_arr = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=np.float32)

        element_probs = GPUVariable(torch.from_numpy(element_probs_arr))
        element_indices = MiniWoBPolicy._sample_elements(element_probs)

        assert np.array_equal(element_indices, [0, 1, 2])
