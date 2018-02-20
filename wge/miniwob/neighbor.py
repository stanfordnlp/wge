import wge.miniwob.distance as distance

from wge.miniwob.state import DOMElementPAD


def potential_neighbors(dom1, dom2):
    """Can be potential neighbors if dom1 and dom2 are not the same element
    and neither are PADs.

    Args:
        dom1 (DOMElement)
        dom2 (DOMElement)

    Returns:
        bool
    """
    # PADs have no neighbors
    if isinstance(dom1, DOMElementPAD) or isinstance(dom2, DOMElementPAD):
        return False

    # Don't be neighbors with yourself
    if dom1.ref == dom2.ref:
        return False

    return True


def is_pixel_neighbor(dom_elem, potential_neighbor, distance_cutoff=30.):
    """Defines if two DOM elements are pixel neighbors of each other. A DOM
    element is a pixel neighbor of another DOM element if they are within 30
    pixels of each other (by default).

    Args:
        dom_elem (DOMElement)
        potential_neighbor (DOMElement)
        distance_cutoff (float): cutoff distance in pixels

    Returns:
        bool: true if dom_elem is a pixel neighbor of potential_neighbor
    """
    if not potential_neighbors(dom_elem, potential_neighbor):
        return False

    dist = distance.row_col_distance(
        dom_elem.left, dom_elem.top,
        dom_elem.left + dom_elem.width,
        dom_elem.top + dom_elem.height,
        potential_neighbor.left, potential_neighbor.top,
        potential_neighbor.left + potential_neighbor.width,
        potential_neighbor.top + potential_neighbor.height)
    return dist <= distance_cutoff


def is_depth_k_lca_neighbor(dom_elem, potential_neighbor, k, cache=None):
    """dom_elem and potential_neighbor are neighbors iff depth(lca(dom_elem,
    potential_neighbor)) >= k

    Args:
        dom_elem (DOMElement)
        potential_neighbor (DOMElement)
        k (int)
        cache (dict((int, int), int)): (ref, ref) --> depth
            optional caches the lca depth of two DOMElements, if provided,
            does lookups and adds to the cache.

    Returns:
        bool
    """
    if not potential_neighbors(dom_elem, potential_neighbor):
        return False

    key = (dom_elem.ref, potential_neighbor.ref)
    if cache is not None:
        if key in cache:
            lca_depth = cache[key]
        else:
            lca_depth = cache[key] = dom_elem.lca(potential_neighbor).depth
    else:
        lca_depth = dom_elem.lca(potential_neighbor).depth
    return lca_depth >= k


def is_text_neighbor(dom_elem, potential_neighbor):
    """Only true if the potential neighbor has text. Useful for pruning
    non-interesting neighbors

    NOTE: is_text_neighbor is NOT symmetric.

    Args:
        dom_elem (DOMElement)
        potential_neighbor (DOMElement)

    Returns:
        bool
    """
    if not potential_neighbors(dom_elem, potential_neighbor):
        return False

    return potential_neighbor.text is not None and \
            potential_neighbor.text != ""
