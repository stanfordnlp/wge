from gtd.ml.vocab import SimpleVocab


class LazyInitVocab(SimpleVocab):
    """Vocab that assigns words to indices on the fly.

    If a word exists in the vocab, word2index returns the associated index,
    otherwise, it assigns the word to a new index if possible up to a max
    size. Once the vocab reaches its max size, all new words are assigned the
    UNK token, which is index 0.

    NOTE: Case sensitive

    Args:
        initial_tokens (list[unicode]): unique list of initial tokens
        max_size (int): maximum size of the vocab
    """
    UNK = u"<unk>"
    def __init__(self, initial_tokens, max_size):
        initial_tokens.insert(0, LazyInitVocab.UNK)
        assert len(initial_tokens) <= max_size
        super(LazyInitVocab, self).__init__(initial_tokens)
        self._max_size = max_size

    # TODO: This is unintuitive and could lead to bugs
    def __len__(self):
        return self._max_size

    def __contains__(self, w):
        """Check if a token has been indexed by this vocab. Adds the token
        into the vocab if possible.
        """
        return self._add_word(w) != self._word2index[LazyInitVocab.UNK]

    def _add_word(self, w):
        """Adds the word to the vocab if possible. Returns the index at which
        it is added. If the word is already in the vocab, just returns its
        index.

        Args:
            w (unicode)

        Returns:
            int (index): 0 if UNK (unsuccessful add)
        """
        # NOTE: Depends on the underlying implementation of SimpleVocab
        # TODO: Make this only depend on interface of SimpleVocab
        if w in self._word2index:
            return self._word2index[w]
        # Add token on the fly
        elif w not in self._word2index and len(self) <= self._max_size:
            index = len(self._index2word)
            self._index2word.append(w)
            self._word2index[w] = index
            return index
        else:
            return self._word2index[LazyInitVocab.UNK]

    def word2index(self, w):
        """Assigns words to indices on the fly. Returns the new index."""
        return self._add_word(w)
