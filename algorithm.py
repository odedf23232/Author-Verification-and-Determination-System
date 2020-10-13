from typing import Optional, Tuple, Dict, Callable

from numpy.core.multiarray import ndarray

import document
import procedures
import distance


def author_verification(document1_chunks, document2_chunks,
                        time_start, time_end, delta_time,
                        distance_function: Callable[[ndarray, ndarray], float] = distance.canberra) \
        -> Tuple[bool, float]:
    """
    Implements the "Author Verification" algorithm, as described in "A Time Series Model
    of the Writing Process", for identifying whether or not two documents where written
    by the same author.

    The verification is done by trying to compare the writing styles used in the two
    documents. The difference in styles is identified by comparing time-series distributions
    of distances (as described in procedures.document_distance) between two "chunks" of the documents,
    with the use of a two-sample test (described in procedures.tst).

    For each jump of the time parameter, from time_start to time_end, a new distribution
    is calculated, where the time parameter indicates the start chunk index of the first
    document from which to sample distance distribution.

    :param document1_chunks: a vector space model representations of all chunks in the first document
    :param document2_chunks: a vector space model representations of all chunks in the second document
    :param time_start: initial delay parameter value
    :param time_end: maximum delay parameter value
    :param delta_time: jump delta for the delay parameter
    :param distance_function: function to calculate distance between 2 chunks
    :return: a tuple holding:
        - whether or not the documents were written by the same author
        - a match percentage, from 0.0 -> 1.0, indicated how much the writing styles match
            A lack of match will mean a low percentage for this value
    """
    # The algorithm according to the article:
    # divide document1 into chunks of size L (length m1)
    # divide document2 into chunks of size L (length m2)
    # concat chunk lists
    # with T from T0 to T* with jump dT:
    #   calculate (Zt,dis,i)
    #       Z1 = Zt from i = T + 1 until m1
    #       Z2 = Zt from i = m1 + 1 until m1 + m2
    #   calculate h = TST(Z1, Z2)
    #   if (h == 0) then same author and we can stop

    all_chunks = document.merge_documents(document1_chunks, document2_chunks)
    document1_chunks_len = document1_chunks.shape[0]
    last_pvalue = -1

    # Running with the time jumps, from start to end with delta jumps
    for time in range(time_start, time_end, delta_time):
        # if time >= document1_chunks_len, then z1, can't be calculated,
        # since it's supposed to be from time until document1_chunks_len
        if time >= document1_chunks_len:
            break

        # calculate distributes z1 and z2
        z1 = procedures.zv_z1(all_chunks, document1_chunks, time, distance_function)
        z2 = procedures.zv_z2(all_chunks, document1_chunks, time, distance_function)

        # We make sure we have enough chunks to actually compare
        # something here
        if len(z1) == 0 or len(z2) == 0:
            break

        # run the two-sample test on the distributions
        h, p_value = procedures.tst(z1, z2)

        if h:
            # there is a match, let's return
            return True, p_value

        # let's save this, so we can return the largest match we found
        last_pvalue = max(last_pvalue, p_value)

    # match was not found
    return False, last_pvalue


def author_determination(document, all_documents, all_authors,
                         time_start, time_end, delta_time,
                         distance_function: Callable[[ndarray, ndarray], float] = distance.canberra) \
        -> Tuple[Optional[str], Dict[int, float]]:
    """
    Implements the "Author Determination" algorithm, as described in "A Time Series Model
    of the Writing Process", for identifying the author of a document, using dataset of known
    documents and authors.

    The determination is done by comparing the document to each known document from the dataset
    with the use of `author_verification` to see if they were written by the same author.

    The author found can only be one from in the dataset, thus if the document was written
    by an author which does not have a documentation in the dataset, it will not be found,
    or simply matched with a wrong author.

    Due to the possibility of false-positives, stemming from similar writing styles of authors and
    the limitation of the transformation function of the documents, the returned author is not
    the first match found, but rather the best match, described with the match percentage returned
    from `author verification`.
    In cases of near-perfect matches (99%), the algorithm will stop comparing for the sake of optimization.

    This implementation uses a time series, to affect the calculation of `author_verification`,
    where `time` is `time_start` passed when running `author_verification`.

    :param document: a vector space model representations of all chunks in the document to find author for
    :param all_documents: a list of all documents in the dataset, where each document is a vector space model
        representations of all chunks in it
    :param all_authors: a list of all authors which have written the documents in the dataset. The list must comply
        with `all_authors[i] author for all_documents[i]` for any `i`.
    :param time_start: initial delay parameter value
    :param time_end: maximum delay parameter value
    :param delta_time: jump delta for the delay parameter
    :param distance_function: function to calculate distance between 2 chunks
    :return: a tuple holding:
        - the name of the author which was determined to have written the document, or None
            if no match was found
        - a dict matching indexes from `all_authors`, to the match percentages they received.
            If the author was never matched by `author_verification`, the percentage will be 0.
    """
    # The algorithm according to the article:
    # with T from T0 to T* with jump dT:
    #   for i to n:
    #       h(i) = author_verification(Document i, Document 0)
    #       if (h(i) == 0) Document 0 was written by author of Document i
    # Unknown author

    # Will hold data about our matches
    positive_matches = {i: 0 for i in range(len(all_authors))}
    stop = False

    # Running with the time jumps, from start to end with delta jumps
    for time in range(time_start, time_end, delta_time):
        # Running on all the documents in the dataset
        for i in range(len(all_documents)):
            # Running author verification to see if same author
            h, match_value = author_verification(all_documents[i], document,
                                                 time, time_end, delta_time,
                                                 distance_function)

            if h:
                # If is same author, we save the score to our matches data, and
                # continue to see if we find better matches.
                positive_matches[i] = max(match_value, positive_matches[i])
                # if the current match percentage is nearly perfect, we can stop
                if match_value >= 0.99:
                    stop = True
                    break

        # If we were told to break this loop, because a really good match was found
        if stop:
            break

    # check if the best score found is 0? if so, then we found no match since 0 is
    # the default value for the dictionary
    best_author_i = max(positive_matches, key=positive_matches.get)
    if positive_matches[best_author_i] == 0:
        return None, positive_matches

    # otherwise, return result with best author
    return all_authors[best_author_i], positive_matches
