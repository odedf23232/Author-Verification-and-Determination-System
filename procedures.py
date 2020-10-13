from typing import Tuple, List, Callable

from numpy.core.multiarray import ndarray
from scipy.sparse import csr_matrix
from scipy.stats import ks_2samp


def document_distance(chunk1: csr_matrix, chunk2: csr_matrix,
                      distance_function: Callable[[ndarray, ndarray], float]) -> float:
    """
    Returns the "distance" between 2 document chunks.

    The distance is described as the difference between the vector space models
    of each chunk, such that the same vectors produce a distance of 0.
    
    :param chunk1: vector space model of the first chunk
    :param chunk2: vector space model of the second chunk
    :param distance_function: function to calculate distance between 2 chunks
    :return: a distance value
    """
    if chunk1.shape[1] != chunk2.shape[1]:
        raise ValueError("vectors should have same length")

    return distance_function(chunk1.toarray(), chunk2.toarray())


def zv_function(chunks: csr_matrix, time: int, i: int,
                distance_function: Callable[[ndarray, ndarray], float]) -> float:
    """
    Calculates the ZV mean function, as described in "A Time Series Model of the Writing Process.pdf" page 4.

    The function is described as a sum of distances between chunk[i] and
    chunks from 1 until `time`, resulting in a Mean relationship between chunk i and
    the set of it's precursors.

    :param chunks: vector space model representation of document chunks
    :param time: delay parameter value
    :param i: chunk index
    :param distance_function: function to calculate distance between 2 chunks
    :return: mean dependency, resulting from the function
    """
    # ZV = (1 / T) * sum(Dis(Di, Di-j))
    # - from "A Time Series Model of the Writing Process.pdf" page 4, formula 1
    sum = 0
    for j in range(time):
        sum += document_distance(chunks[i], chunks[i - j], distance_function)

    return (1 / time) * sum


def zv_z1(all_chunks: csr_matrix, document: csr_matrix, time: int,
          distance_function: Callable[[ndarray, ndarray], float]) -> List[float]:
    """
    Calculates distribution of mean dependency resulting from `zv_function`, for
    the first document in `all_chunks`, as described in "A Time Series Model of the Writing Process.pdf" page 5, part 6.

    :param all_chunks: combination of chunks from 2 documents, stacked upon each other
    :param document: first document in the combination
    :param time: delay parameter
    :param distance_function: function to calculate distance between 2 chunks
    :return: list of results from `zv_function` starting from time+1 until len(document)
    """
    return [zv_function(all_chunks, time, i, distance_function)
            for i in range(time + 1, document.shape[0])]


def zv_z2(all_chunks: csr_matrix, document: csr_matrix, time: int,
          distance_function: Callable[[ndarray, ndarray], float]) -> List[float]:
    """
    Calculates distribution of mean dependency resulting from `zv_function`, for
    the second document in `all_chunks`, as described in "A Time Series Model of the Writing Process.pdf"
    page 5, part 6.

    :param all_chunks: combination of chunks from 2 documents, stacked upon each other
    :param document: first document in the combination
    :param time: delay parameter
    :param distance_function: function to calculate distance between 2 chunks
    :return: list of results from `zv_function` starting from len(document) until len(all_chunks)
    """
    return [zv_function(all_chunks, time, i, distance_function)
            for i in range(document.shape[0] + 1, all_chunks.shape[0])]


def tst(z1: List[float], z2: List[float]) -> Tuple[bool, float]:
    """
    Performs a Two-Sample test between two term-frequency distributions, as result from
    `zt_function`. The test describes the difference or similarity between the two distributions.

    The Kolmogorov–Smirnov two-sample test is used for the calculation, where the results
    describes a percentage (from 0 to 1) of match. If the percentage is higher than 0.05
    we can awesome a positive match.

    :param z1: distributions from `zv_z1` function
    :param z2: distributions from `zv_z2` function
    :return: a tuple holding:
        - whether or not a match exists
        - the match percentage
    """
    # H 0 : F (x) = G(x)
    # a T ST procedure in an algorithm which returns 1 if H 0 is rejected and 0 otherwise
    # Thus, the lower your p value the greater the statistical evidence you have to reject the
    # null hypothesis and conclude the distributions are different.
    # - from "A Time Series Model of the Writing Process.pdf" page 6
    # First value is the test statistics, and second value is the p-value.
    # if the p-value is less than 95 (for a level of significance of 5%), this means that you
    # cannot reject the Null-hypothesis that the two sample distributions are identical
    # - from
    # https://stackoverflow.com/questions/39132469/how-to-interpret-scipy-stats-kstest-and-ks-2samp-to-evaluate-fit-of-data-t
    _, p_value = ks_2samp(z1, z2)
    return p_value >= 0.05, p_value
