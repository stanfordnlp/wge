from wge.miniwob.distance import rectangle_distance, \
        line_segment_distance


class TestDistanceModule(object):
    def test_rectangle_distance(self):
        pass

    def test_line_segment_distance(self):
        # six cases of ((s1, e1, s2, e2), dist)
        cases = [
            ((0, 1, 2, 3), 1), # (s1, e1, s2, e2)
            ((0, 2, 1, 3), 0), # (s1, s2, e1, e2)
            ((0, 3, 1, 2), 0), # (s1, s2, e1, e1)
            ((1, 2, 0, 3), 0), # (s2, s1, e1, e2)
            ((2, 3, 0, 1), 1), # (s2, e2, s1, e1)
            ((1, 3, 0, 2), 0), # (s2, s1, e2, e1)
        ]

        for (s1, e1, s2, e2), dist in cases:
            assert dist == line_segment_distance(s1, e1, s2, e2)
