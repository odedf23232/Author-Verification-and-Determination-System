import algorithm
import datasets
import document
import distance

DOCUMENT_CHUNK_SIZE = 7000
TIME_START = 10
TIME_END = 100
DELTA_TIME = 10

dataset = datasets.dataset_local_books()
authors, documents, _ = document.load_documents_from_dataset(DOCUMENT_CHUNK_SIZE, dataset)
print('Data ready')

index = 0
document_to_test = documents[index]

documents_train = [documents[i] for i in range(len(documents)) if i != index]
authors_train = [authors[i] for i in range(len(documents)) if i != index]
author, all_matches = algorithm.author_determination(document_to_test,
                                                     documents_train, authors_train,
                                                     TIME_START, TIME_END, DELTA_TIME)
print(author, "-", authors[index])
print(all_matches)
