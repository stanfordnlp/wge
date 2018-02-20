class Cache(object):
    """Provides a set of APIs for caching."""
    def __init__(self):
        self._cache = dict()

    def uncached_keys(self, keys):
        """Returns a list of the keys that are not cached in the same order as
        keys.

        Args:
            keys (list[Hashable])

        Returns:
            uncached_keys (list[Hashable])
        """
        uncached_keys = []
        for key in keys:
            if key not in self._cache:
                uncached_keys.append(key)
        return uncached_keys

    def get(self, keys):
        """Given a set of cached keys, returns associated cached values.

        Args:
            keys (list[Hashable])

        Returns:
            values (list[Object]): in same order as keys
        """
        return [self._cache[key] for key in keys]

    def cache(self, keys, values):
        """Associates the keys with the values.

        Args:
            keys (list[Hashable])
            values (list[Object]): same length as keys
        """
        assert len(keys) == len(values)

        for key, value in zip(keys, values):
            self._cache[key] = value

    def clear(self):
        """Clears the cache."""
        self._cache = dict()
