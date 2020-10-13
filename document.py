import string
from typing import List, Tuple

from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def _get_stop_words() -> List[str]:
    """
    Loads stop words for using when transforming.
    :return: list of stop words.
    """
    # In "A Time Series Model of the Writing Process.pdf" page 8,
    # it is specified that stop words from the following link were used:
    # from https://www.ranks.nl/stopwords -> Long Stopword List
    with open('stopwords.txt') as f:
        return f.read().split('\n')


# a global stop_words variable, so that we don't load it each time
_stop_words = _get_stop_words()


def _text_process(text: str) -> List[str]:
    """
    Processor for finding words the interest us in each document.

    :param text: document text
    :return: list of words/terms which interest us when transforming
    """
    # Removal of Punctuation Marks
    text = ''.join([char for char in text if char not in string.punctuation])
    # Removal of anything that is not a stop word
    # Any uppercase characters in the texts involved in the experiments are con-
    # verted to the corresponding lowercase characters, and all other characters are
    # unchanged
    # - from "A Time Series Model of the Writing Process.pdf" page 8
    text = text.lower()
    # In this paper, a Vector Space Model is built, resting upon the content-free
    # words and the stop words. The joint occurrences of the content-free words can
    # provide valuable stylistic evidence for authorship
    # - from "A Time Series Model of the Writing Process.pdf" page 8
    return [word for word in text.split() if word in _stop_words]


def _transform(texts: List[str], chunk_size: int) -> Tuple[List[csr_matrix], List[str]]:
    """
    Transforms all the documents into vector space models.

    The transformation shows each document as a collection of chunks of a given size,
    where each chunk is a vector-space model. The chunk models are Term-Frequency
    Inverse-Document-Frequency vectors, showing the frequency of term usages in the document.

    Most terms from the document are stripped, such that only stop-words and content-free
    terms remain, since they can be used to describe a writing style.

    :param texts: list of documents to transform
    :param chunk_size: length of each chunk, in character count
    :return: a tuple holding:
        - list of the transformed documents
        - list of terms in the TF-IDF vectors
    """
    # What we do here is a bit premature, as the chunks are needed specifically
    # for use in author verification algorithm which require dividing the documents
    # into chunks.
    # Instead of doing this each time, and then having to transform the chunks
    # into vector space models, which will take some work...
    # We just do this here once for all the documents, and represent
    # each as a csr_matrix, where each row of the matrix is a vector representing
    # a chunk

    # Divide each documents into chunks
    chunks = [[text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
              for text in texts]

    # Transform each document chunks using TFIDF
    # We use the raw texts from the documents as the "training data" for the
    # vectorizer. From those texts, we extract terms using `_text_process`,
    # and map the frequency of those terms in each chunk.
    transformer = TfidfVectorizer(use_idf=False, analyzer=_text_process).fit(texts)
    chunks = [transformer.transform(doc_chunks) for doc_chunks in chunks]

    return chunks, transformer.get_feature_names()


def load_documents_from_dataset(chunk_size: int, dataset: Tuple[List[str], List[str]]) \
        -> Tuple[List[str], List[csr_matrix], List[str]]:
    """
    Loads and transforms a dataset into documents for the algorithm

    Each document in the dataset is divided into chunks of a constant size,
    and each chunk is transformed into a vector space model which is used to
    describe the writing style used.

    :param chunk_size: length of each chunk, in character count
    :param dataset: a tuple of list of authors to list of documents, where `authors[i] is author of documents[i]`.
    :return: a tuple holding:
        - list of authors from the dataset
        - list of transformed documents, where `authors[i] is author of documents[i]`,
            and each document is made up of several chunks of a given size
        - list of words which are described in the model of each documents chunk
    """
    authors, texts = dataset
    chunks, feature_names = _transform(texts, chunk_size)
    return authors, chunks, feature_names


def merge_documents(document1_chunks: csr_matrix, document2_chunks: csr_matrix) -> csr_matrix:
    """
    Merges two documents into one matrix, where the resulting matrix
    will contain the rows of document2 bellow the rows of document1, such
    that the length of the resulting matrix is `len(document1) + len(document2)`.

    :param document1_chunks: a vector space model representations of all chunks in the first document
    :param document2_chunks: a vector space model representations of all chunks in the second document
    :return: a new document matrix
    """
    return sparse.vstack((document1_chunks, document2_chunks), format='csr')
