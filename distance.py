from numpy.core.multiarray import ndarray
from scipy.stats import spearmanr
import scipy.spatial.distance as scipy_distance


def spearman(chunk1: ndarray, chunk2: ndarray) -> float:
    """
    Returns the "distance" between 2 document chunks.

    The distance is described as the difference between the vector space models
    of each chunk, such that the same vectors produce a distance of 0.

    We use Spearman's correlation.

    :param chunk1: vector space model of the first chunk
    :param chunk2: vector space model of the second chunk
    :return: a distance value
    """
    # - from "A Time Series Model of the Writing Process.pdf" page 7, sub 1
    result = spearmanr(chunk1[chunk2 != 0], chunk2[chunk2 != 0])
    return 1 - result.correlation


def euclidean(chunk1: ndarray, chunk2: ndarray) -> float:
    """
    Returns the "distance" between 2 document chunks.

    The distance is described as the difference between the vector space models
    of each chunk, such that the same vectors produce a distance of 0.

    We use the Euclidean distance algorithm between 2 vectors in an n-dimensional vector
    space.

    :param chunk1: vector space model of the first chunk
    :param chunk2: vector space model of the second chunk
    :return: a distance value
    """
    # - from "A Time Series Model of the Writing Process.pdf" page 8, sub 2
    return scipy_distance.euclidean(chunk1, chunk2)


def canberra(chunk1: ndarray, chunk2: ndarray) -> float:
    """
    Returns the "distance" between 2 document chunks.

    The distance is described as the difference between the vector space models
    of each chunk, such that the same vectors produce a distance of 0.

    We use the Canberra distance algorithm between 2 vectors in an n-dimensional vector
    space.

    :param chunk1: vector space model of the first chunk
    :param chunk2: vector space model of the second chunk
    :return: a distance value
    """
    # C(P, Q) = sum[i->infinity](abs(Pi - Qi) / (abs(Pi) - abs(Qi)))
    # - from "A Time Series Model of the Writing Process.pdf" page 8, sub 3
    # The Canberra distance d between vectors p and q in an n-dimensional real vector space
    # where p, q are vectors of length n
    return scipy_distance.canberra(chunk1, chunk2)
