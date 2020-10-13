import itertools

import pandas
from scipy import sparse

import algorithm
import document
import procedures


def document_term_frequency(document_chunks, feature_names):
    return pandas.DataFrame.sparse.from_spmatrix(document_chunks,
                                                 columns=feature_names)


def zv_distributions(chunks, time, i_start, i_end, column_name='zv'):
    results = [procedures.zv_function(chunks, time, i) for i in range(i_start, i_end)]
    return pandas.DataFrame(results,
                            index=[i for i in range(i_start, i_end)], columns=[column_name])


def multiple_zv_distributions(documents, time, i_start, i_end):
    frames = []
    for i in range(len(documents)):
        for j in range(len(documents)):
            name = 'docs[{},{}]'.format(str(i), str(j))
            all_chunks = document.merge_documents(documents[i], documents[j])
            df = zv_distributions(all_chunks, time, i_start, i_end, column_name=name) \
                .rename_axis(name, axis=1)
            frames.append(df)

    return frames


def collection_av_run_data(documents, time_start, time_end, delta_time, distance_function):
    results = []
    for i in range(len(documents)):
        for j in range(len(documents)):
            result, p_value = algorithm.author_verification(documents[i], documents[j],
                                                            time_start, time_end, delta_time,
                                                            distance_function)
            results.append((i, j, p_value, result))

    return results


def p_value_document_distribution(documents, authors, av_run_data):
    data_pvalues = [[None for j in range(len(documents))] for i in range(len(documents))]
    for data in av_run_data:
        data_pvalues[data[0]][data[1]] = data[2]

    return pandas.DataFrame.sparse.from_spmatrix(sparse.csr_matrix(data_pvalues),
                                                 index=authors, columns=authors)


def p_value_to_match_distribution(av_run_data):
    data = [[int(data[3]), data[2]] for data in av_run_data]
    return pandas.DataFrame(data, columns=['Found', 'P Value'])


def collect_run_data(documents, authors, time_start, time_end, delta_time, distance_function):
    av_results = []
    ad_results = []

    for i in range(len(documents)):
        for j in range(len(documents)):
            result, p_value = algorithm.author_verification(documents[i], documents[j],
                                                            time_start, time_end,
                                                            delta_time, distance_function)
            av_results.append((i, j, p_value, result))

    for i in range(len(documents)):
        o_documents = [documents[j] for j in range(len(documents)) if j != i]
        o_authors = [authors[j] for j in range(len(documents)) if j != i]
        author, other = algorithm.author_determination(documents[i], o_documents, o_authors,
                                                       time_start, time_end, delta_time,
                                                       distance_function)
        ad_results.append((author, other))

    return av_results, ad_results


def av_match_table(documents, authors, av_run_data):
    data_match = [[None for j in range(len(documents))] for i in range(len(documents))]
    for data in av_run_data:
        data_match[data[0]][data[1]] = int(data[3])

    return pandas.DataFrame.sparse.from_spmatrix(sparse.csr_matrix(data_match),
                                                 index=authors, columns=authors)


def ad_result_table(authors, ad_run_data):
    match_score = [max(d[1].values()) for d in ad_run_data]
    deter_results = [ad_run_data[i][0] if match_score[i] > 0 else 'None'
                     for i in range(len(ad_run_data))]
    is_correct = [authors[i] == deter_results[i] for i in range(len(authors))]

    return pandas.DataFrame({'Result': deter_results, 'Author': authors,
                             'Is Correct': is_correct, 'Score': match_score},
                            columns=['Author', 'Result', 'Is Correct', 'Score'])


def av_result_table(authors, av_run_data):
    authors1 = []
    authors2 = []
    results = []
    score = []
    for d in av_run_data:
        authors1.append(authors[d[0]])
        authors2.append(authors[d[1]])
        score.append(d[2])
        results.append(d[3])

    return pandas.DataFrame({'Result': results,
                             'Author1': authors1, 'Author2': authors2,
                             'Score': score},
                            columns=['Author1', 'Author2', 'Result', 'Score'])


import distance
import json
import datasets
import document


def run_collection_distance_sizes(dataset):
    distance_functions = {
        'spearman': distance.spearman,
        'euclidean': distance.euclidean,
        'canberra': distance.canberra
    }
    chunk_ranges = range(2000, 10000, 1000)
    time_ranges = [(10, 50, 10), (10, 100, 20), (100, 200, 10)]

    for chunk_size in chunk_ranges:
        authors, documents, feature_names = document.load_documents_from_dataset(chunk_size, dataset)

        for time_start, time_end, delta_time in time_ranges:
            for name, distance_func in distance_functions.items():
                print('Running for func', name)
                av = collection_av_run_data(documents,
                                            time_start, time_end, delta_time,
                                            distance_func)
                av_result_table(authors, av) \
                    .to_csv('results/results_av_{}-{}-{}_{}_{}.csv'.format(
                        str(time_start), str(time_end), str(delta_time),
                        str(chunk_size), name))
