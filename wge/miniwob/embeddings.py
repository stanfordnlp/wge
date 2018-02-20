import torch

import numpy as np
import torch.nn.functional as F
import wge.miniwob.positions as positions
import wge.miniwob.neighbor as N

from collections import namedtuple
from gtd.ml.torch.seq_batch import SequenceBatch
from gtd.ml.torch.token_embedder import TokenEmbedder
from gtd.ml.torch.utils import GPUVariable
from gtd.ml.vocab import SimpleEmbeddings, SimpleVocab, Vocab
from gtd.utils import flatten, as_batches
from torch.nn import Module, Conv2d, Linear, Parameter
from wge.embeddings import Embedder, BoolEmbeddings, LazyInitEmbeddings
from wge.miniwob.state import DOMElementPAD
from wge.utils import word_tokenize
from wge.vocab import LazyInitVocab


class ExtendedTagVocab(SimpleVocab):
    """Vocab object for all HTML tags. Note that is also supports different
    types of input tags input_text, input_radio, input_checkbox, and t for
    text.
    """

    def __init__(self):
        tags = ["!--...--", "!DOCTYPE", "a", "abbr", "acronym", "address",
                "applet", "area", "article", "aside", "audio", "b", "base",
                "basefont", "bdi", "bdo", "big", "blockquote", "body", "br",
                "button", "canvas", "caption", "center", "cite", "code",
                "col", "colgroup", "datalist", "dd", "del", "details", "dfn",
                "dialog", "dir", "div", "dl", "dt", "em", "embed", "fieldset",
                "figcaption", "figure", "font", "footer", "form", "frame",
                "frameset", "h1", "h2", "h3", "h4", "h5", "h6", "head",
                "header", "hr", "html", "i", "iframe", "img", "ins", "kbd",
                "keygen", "label", "legend", "li", "link", "main", "map",
                "mark", "menu", "menuitem", "meta", "meter", "nav",
                "noframes", "noscript", "object", "ol", "optgroup", "option",
                "output", "p", "param", "picture", "pre", "progress", "q",
                "rp", "rt", "ruby", "s", "samp", "script", "section",
                "select", "small", "source", "span", "strike", "strong",
                "style", "sub", "summary", "sup", "table", "tbody", "td",
                "textarea", "tfoot", "th", "thead", "time", "title", "tr",
                "track", "tt", "u", "ul", "var", "video", "wbr"]
        tags.extend('input_' + x for x in [
            "button", "checkbox", "color", "date", "datetime", "datetime-local",
            "email", "file", "hidden", "image", "month", "number", "password",
            "radio", "range", "reset", "search", "submit", "tel", "text",
            "time", "url", "week"])
        tags.append("t")
        super(ExtendedTagVocab, self).__init__(tags)


class TagEmbeddings(SimpleEmbeddings):
    """Embeds the tag of DOM elements."""

    def __init__(self, embed_dim):
        tags = ExtendedTagVocab()
        embed_matrix = np.random.uniform(
            -np.sqrt(3. / embed_dim), np.sqrt(3. / embed_dim),
            size=(len(tags), embed_dim)).astype(np.float32)
        super(TagEmbeddings, self).__init__(embed_matrix, tags)


class DOMValueVocab(SimpleVocab):
    """Vocab for the values that DOM elements can take."""

    def __init__(self):
        # TODO: Add other values
        values = [True, False, None, ""]
        super(DOMValueVocab, self).__init__(values)


class DOMValueEmbeddings(SimpleEmbeddings):
    """Embeds the value of DOM elements."""

    def __init__(self, embed_dim):
        values = DOMValueVocab()
        embed_matrix = np.random.uniform(
            -np.sqrt(3. / embed_dim), np.sqrt(3. / embed_dim),
            size=(len(values), embed_dim)).astype(np.float32)
        super(DOMValueEmbeddings, self).__init__(embed_matrix, values)


class DOMAlignments(Module):
    """Computes the alignments between DOM elements and Fields.

    If a DOM element (d) aligns with the i_1th, ..., i_nth values of the
    field (f.values[i_1], ..., f.values[i_n]), the corresponding vector for d
    is sum(e[keys2index(f.keys[i_1])], ..., (e[keys2index(f.keys[i_n])))
    where e is an embedding matrix

    keys2index guarantees unique indices of num_buckets keys, any additional
    keys will be lumped in the same buckets as other keys.

    Alignments defined by d.text != "" and d.text in f.values[i]

    Args:
        embed_dim (int): the embedding dim of the alignments
        num_buckets (int): max size of embedding matrix e
    """
    def __init__(self, embed_dim, num_buckets=20):
        super(DOMAlignments, self).__init__()
        self._embed_dim = embed_dim

        # num_bucket unique spots, 1 spot for rest
        self._num_buckets = num_buckets + 1
        self._keys2index = LazyInitVocab([], self._num_buckets)

        # Index 0 for not align, index 1 for align
        max_val = np.sqrt(3. / embed_dim)
        self._alignment_embeds = Parameter(
                torch.Tensor(self._num_buckets, embed_dim).uniform_(
                    -max_val, max_val))

    def forward(self, dom_elements, alignment_fields):
        """Computes the alignments. An element aligns iff elem.text
        in utterance and elem.text != ""

        Args:
            dom_elements (list[list[DOMElement]]): batch of set of DOM
                elements (padded to be unragged)
            alignment_fields (list[Fields]): batch of fields. Alignments
                computed with the values of the fields.

        Returns:
            Variable[FloatTensor]: batch x num_elems x embed_dim
                The aligned embeddings per DOM element
        """
        batch_size = len(dom_elements)
        assert batch_size > 0
        num_dom_elems = len(dom_elements[0])
        assert num_dom_elems > 0

        # mask batch_size x num_dom_elems x num_buckets
        alignments = np.zeros(
            (batch_size, num_dom_elems, self._num_buckets)).astype(np.float32)

        # Calculate the alignment matrix between elems and fields
        for batch_idx in xrange(len(dom_elements)):
            for dom_idx, dom in enumerate(dom_elements[batch_idx]):
                keys = alignment_fields[batch_idx].keys
                vals = alignment_fields[batch_idx].values
                for key, val in zip(keys, vals):
                    if dom.text and dom.text in val:
                        align_idx = self._keys2index.word2index(key)
                        alignments[batch_idx, dom_idx, align_idx] = 1.

        # Flatten alignments for SequenceBatch
        # (batch * num_dom_elems) x num_buckets
        alignments = GPUVariable(torch.from_numpy(alignments.reshape(
                (batch_size * num_dom_elems, self._num_buckets))))

        # (batch * num_dom_elems) x num_buckets x embed_dim
        expanded_alignment_embeds = self._alignment_embeds.expand(
                batch_size * num_dom_elems,
                self._num_buckets, self.embed_dim)
        alignment_seq_batch = SequenceBatch(
                expanded_alignment_embeds, alignments, left_justify=False)

        # (batch * num_dom_elems) x alignment_embed_dim
        alignment_embeds = SequenceBatch.reduce_sum(alignment_seq_batch)
        return alignment_embeds.view(
                batch_size, num_dom_elems, self.embed_dim)

    @property
    def embed_dim(self):
        return self._embed_dim


class BaseDOMElementEmbedder(Embedder):
    """Embeds a single DOMElement based on its text, tag and value."""

    def __init__(self, utterance_embedder, tag_embed_dim,
                 value_embed_dim, tampered_embed_dim, classes_embed_dim,
                 max_classes=2000, max_tags=100):
        """
        Args:
            utterance_embedder (UtteranceEmbedder)
            tag_embed_dim (int): embedding dim of tags
            tampered_embed_dim (int): embedding dim of the tampered attribute
            classes_embed_dim (int): embedding dim of the classes
            max_classes (int): maximum number of supported classes to embed
        """
        super(BaseDOMElementEmbedder, self).__init__()

        self._utterance_embedder = utterance_embedder
        self._tag_embedder = TokenEmbedder(
            LazyInitEmbeddings(tag_embed_dim, max_tags), trainable=True)
        self._value_embedder = TokenEmbedder(
            DOMValueEmbeddings(value_embed_dim), trainable=True)
        self._tampered_embedder = TokenEmbedder(
            BoolEmbeddings(tampered_embed_dim), trainable=True)
        self._classes_embedder = TokenEmbedder(
            LazyInitEmbeddings(classes_embed_dim, max_classes), trainable=True)
        self._colors_dim = 8  # 4 (rgba) for fg and 4 for bg
        self._coords_dim = 2  # left and top

    @classmethod
    def from_config(cls, utterance_embedder, config):
        """Constructs a BaseDOMElementEmbedder from a config.

        Args:
            utterance_embedder (UtteranceEmbedder): the utterance embedder
            config (Config): has tag_embed_dim, value_embed_dim,
                tampered_embed_dim, classes_embed_dim

        Returns:
            BaseDOMElementEmbedder
        """
        return cls(utterance_embedder, config.tag_embed_dim,
                   config.value_embed_dim, config.tampered_embed_dim,
                   config.classes_embed_dim)

    def forward(self, dom_elem):
        """Embeds a batch of DOMElements.

        Args:
            dom_elem (list[list[DOMElement]]): batch of list of DOM. Each
                batch must already be padded to have the same number of DOM
                elements.

        Returns:
            Variable(FloatTensor): batch x num_dom_elems x embed_dim
        """
        # Check that the batches are rectangular
        for dom_list in dom_elem:
            assert len(dom_list) == len(dom_elem[0])

        num_dom_elems = len(dom_elem[0])
        dom_elem = flatten(dom_elem)

        # (batch * max_dom_num) x lstm_dim
        text_embeddings = []
        for batch in as_batches(dom_elem, 100):
            final_states, combined_states = self._utterance_embedder(
                [word_tokenize(dom.text) for dom in batch])
            text_embeddings.append(final_states)
        text_embeddings = torch.cat(text_embeddings, 0)

        # (batch * max_dom_num) x tag_embed_dim
        tag_embeddings = self._tag_embedder.embed_tokens(
            [dom.tag for dom in dom_elem])

        value_embeddings = self._value_embedder.embed_tokens(
            [bool(dom.value) for dom in dom_elem])

        tampered_embeddings = self._tampered_embedder.embed_tokens(
            [dom.tampered for dom in dom_elem])

        class_embeddings = self._classes_embedder.embed_tokens(
            [dom.classes for dom in dom_elem])

        # (batch * max_dom_num) x 4
        fg_colors = [GPUVariable(
            torch.FloatTensor(elem.fg_color)) for elem in dom_elem]
        fg_colors = torch.stack(fg_colors)
        bg_colors = [GPUVariable(
            torch.FloatTensor(elem.bg_color)) for elem in dom_elem]
        bg_colors = torch.stack(bg_colors)

        # (batch * max_dom_num) x 2
        coords = [GPUVariable(torch.FloatTensor(
            (float(elem.left) / positions.IMAGE_COLS,
             float(elem.top) / positions.IMAGE_ROWS)))
            for elem in dom_elem]
        coords = torch.stack(coords)

        # (batch * max_dom_num) * dom_embed_dim
        dom_embeddings = torch.cat(
            (text_embeddings, tag_embeddings, value_embeddings,
             tampered_embeddings, class_embeddings, coords, fg_colors,
             bg_colors), dim=1)

        # batch x max_dom_num x dom_embed_dim
        return dom_embeddings.view(-1, num_dom_elems, self.embed_dim)

    @property
    def embed_dim(self):
        return self._tag_embedder.embed_dim + \
               self._utterance_embedder.embed_dim + \
               self._value_embedder.embed_dim + \
               self._tampered_embedder.embed_dim + \
               self._colors_dim + self._coords_dim + \
               self._classes_embedder.embed_dim


class SuperSimpleVocab(Vocab):
    """Even simpler than SimpleVocab.

    Currently only used in HigherOrderDOMElementEmbedder.

    Doesn't perform any checks, e.g.:
    - tokens don't have to be unique
    - tokens don't have to be unicode values
    """

    def __init__(self, tokens):
        t2i = {}
        i2t = {}
        for i, token in enumerate(tokens):
            t2i[token] = i
            i2t[i] = token
        self.token2index = t2i
        self.index2token = i2t

    def word2index(self, w):
        return self.token2index[w]

    def index2word(self, i):
        return self.index2token[i]


class HigherOrderDOMElementEmbedder(Embedder):
    def __init__(self, base_embed_dim):
        # needs to come first
        super(HigherOrderDOMElementEmbedder, self).__init__()
        self._base_embed_dim = base_embed_dim

        # Optimization: cache the lca computations (ref based)
        # ASSUMPTION: refs uniquely identify a DOM tree position, so that
        # refs unique identify an LCA across all episodes
        # WARNING: Could possibly lead to difficult bugs if the assumption is
        # violated
        self._lca_cache = {}
        self._lca_depth_start = 3  # Depth to start embedding LCAs
        self._lca_depth_end = 7  # EXCLUSIVE on top [3, 6)

        lca_range = self._lca_depth_end - self._lca_depth_start

        # Make the total DOM tree neighbors embed dim to be approximately the
        # base embed dim
        projection_dim = base_embed_dim / lca_range
        self._dom_neighbor_projection = Linear(
            base_embed_dim, projection_dim)
        self._dom_neighbors_embed_dim = projection_dim * lca_range

    @property
    def embed_dim(self):
        return 2 * self._base_embed_dim + self._dom_neighbors_embed_dim

    def forward(self, dom_elems, base_dom_embeds):
        """Embeds a batch of DOMElement sequences by mixing base embeddings on
        notions of neighbors.

        Args:
            dom_elems (list[list[DOMElement]]): a batch of DOMElement
                sequences to embed. All sequences must be padded to have the
                same number of DOM elements.
            base_dom_embeds (Variable[FloatTensor]):
                batch_size, num_dom_elems, base_dom_embed_dim

        Returns:
            dom_embeds (Variable[FloatTensor]): of shape (batch_size,
                num_dom_elems, embed_dim)
        """
        batch_size, num_dom_elems, embed_dim = base_dom_embeds.size()

        # flatten, for easier processing
        base_dom_embeds_flat = base_dom_embeds.view(
            batch_size * num_dom_elems, embed_dim)

        # list of length: batch_size * num_dom_elems
        dom_elems_flat = flatten(dom_elems)
        assert len(dom_elems_flat) == batch_size * num_dom_elems

        # DOM neighbors whose LCA goes from depth 3 to 6 (root is depth 1)
        dom_neighbor_embeds = []
        for k in xrange(self._lca_depth_start, self._lca_depth_end):
            is_neighbor_fn = lambda elem1, elem2: (N.is_depth_k_lca_neighbor(
                elem1, elem2, k, self._lca_cache) and
                                                   N.is_text_neighbor(elem1,
                                                                      elem2))
            dom_neighbor_indices = self._get_neighbor_indices(
                dom_elems, is_neighbor_fn)

            # TODO: reduce_max
            # (batch_size * num_dom_elems, embed_dim)
            neighbor_embedding = SequenceBatch.reduce_sum(
                SequenceBatch.embed(
                    dom_neighbor_indices, base_dom_embeds_flat))
            # TODO: Batch these projections? For performance
            projected_neighbor_embedding = self._dom_neighbor_projection(
                neighbor_embedding)
            dom_neighbor_embeds.append(projected_neighbor_embedding)

        # (batch_size * num_dom_elems, lca_range * (base_embed_dim /
        # lca_range))
        dom_neighbor_embeds = torch.cat(dom_neighbor_embeds, 1)

        # SequenceBatch of shape (batch_size * num_dom_elems, max_neighbors)
        pixel_neighbor_indices = self._get_neighbor_indices(
            dom_elems,
            lambda elem1, elem2: (N.is_pixel_neighbor(elem1, elem2) and
                                  N.is_text_neighbor(elem1, elem2)))

        # SequenceBatch of shape
        # (batch_size * num_dom_elems, max_neighbors, embed_dim)
        pixel_neighbor_embeds = SequenceBatch.embed(
            pixel_neighbor_indices, base_dom_embeds_flat)

        # TODO(kelvin): switch to reduce_max
        # (batch_size * num_dom_elems, embed_dim)
        pixel_neighbor_embeds_flat = SequenceBatch.reduce_mean(
            pixel_neighbor_embeds, allow_empty=True)

        dom_embeds_flat = torch.cat(
            [base_dom_embeds_flat, pixel_neighbor_embeds_flat,
             dom_neighbor_embeds], 1)
        dom_embeds = dom_embeds_flat.view(
            batch_size, num_dom_elems, self.embed_dim)

        return dom_embeds

    def _get_neighbor_indices(self, dom_elements, is_neighbor):
        """Compute neighbor indices.

        Args:
            dom_elements (list[DOMElement]): may include PAD elements
            is_neighbor (Callable: DOMElement x DOMElement --> bool): True if
                two DOM elements are neighbors of each other, otherwise False

        Returns:
            SequenceBatch: of shape (total_dom_elems, max_neighbors)
        """
        dom_element_ids = [id(e) for e in flatten(dom_elements)]
        dom_element_ids_set = set(dom_element_ids)
        vocab = SuperSimpleVocab(dom_element_ids)

        neighbors_batch = []
        for dom_batch in dom_elements:
            for dom_elem in dom_batch:
                # Optimization: no DOM PAD has neighbors
                if isinstance(dom_elem, DOMElementPAD):
                    neighbors = []
                else:
                    neighbors = []
                    for neighbor in dom_batch:
                        if is_neighbor(dom_elem, neighbor):
                            neighbors.append(id(neighbor))

                neighbors_batch.append(neighbors)

        neighbor_indices = SequenceBatch.from_sequences(
            neighbors_batch, vocab, min_seq_length=1)
        return neighbor_indices


class DOMContextEmbedder(Embedder):
    """Takes a batch of DOM embeddings and a batch of utterance embedding and
    constructs a context vector from the two.

    Args:
        input_dim (int): the dim of dom embed + dim of utt embed
        output_dim (int): the dim of the context vector
    """

    def __init__(self, input_dim, output_dim):
        super(DOMContextEmbedder, self).__init__()
        self._linear = Linear(input_dim, output_dim, bias=True)
        self._embed_dim = output_dim

    def forward(self, dom_embeds, utt_embeds):
        """Creates a context vector.

        Args:
            dom_embeds (Variable[FloatTensor]): batch x dom_embed_dim
            utt_embeds (Variable[FloatTensor]): batch x utt_embed_dim

        Returns:
            context (Variable[FloatTensor]): batch x output_dim
        """
        concat = torch.cat([dom_embeds, utt_embeds], 1)
        context = F.relu(self._linear(concat))
        return context

    @property
    def embed_dim(self):
        return self._embed_dim
