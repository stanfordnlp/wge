class Mask(object):
    """Filters out certain indices from a list

    on __init__: given a list with 0 or more None, creates a mask that filters
        out indices corresponding to the None indices in the list
    on filter: given a list, returns the list with the mask indices removed
    on insert_none: given a list, returns the list with None inserted
        at the mask indices

    Args:
        mask_creator (list[]): list with 0 or more None
    """
    def __init__(self, mask_creator):
        self._len = len(mask_creator)
        self._none_indices = [i for i, obj in enumerate(
            mask_creator) if obj is None]

    def filter(self, to_filter):
        """Returns to_filter with the mask indices removed
        
        Args:
            to_filter (list[]): must have same len as mask_creator

        Returns:
            list[]
        """
        assert(len(to_filter) == self._len)
        filtered = []
        for i, thing in enumerate(to_filter):
            if i not in self._none_indices:
                filtered.append(thing)
        return filtered

    def insert_none(self, list_to_insert):
        """Inserts None at the indices specified by the list that this mask
        was constructed on.

        Args:
            list_to_insert (list[]): a list of things to insert None into.
                Must get padded to the same length as original

        Returns:
            list[]: same list with None at the right indices
        """
        assert(self._len - len(self._none_indices) == len(list_to_insert))

        for index in self._none_indices:
            list_to_insert.insert(index, None)
        return list_to_insert
