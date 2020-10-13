import json
from typing import List, Tuple

from newspaper import article

import pdf


def dataset_web_articles(amount_limit: int = -1) -> Tuple[List[str], List[str]]:
    """
    Loads a dataset of news articles from the web.
    The articles are online and must be downloaded and parsed.

    :param amount_limit: limits the amount of articles. Non-positive if to ignore
        and get everything. If bigger than the amount of articles, then
        just returns all the articles.
    :return: list of authors, list of documents
    """
    with open('data/News_Category_Dataset_v2.json') as f:
        data = json.load(f)

    if amount_limit > 0:
        data = data[:max(amount_limit, len(data))]

    authors = []
    texts = []
    for d in data:
        author = d['authors']

        try:
            artic = article.Article(d['link'])
            artic.download()
            artic.parse()
            texts.append(artic.text)
            authors.append(author)
        except Exception as e:
            print('Error downloading', str(e))

    return authors, texts


def dataset_local_books():
    """
    Loads a dataset of books from local files.
    The books are in PDF format, and are converted to raw text.

    :return: list of authors, list of documents
    """
    with open('data/books_index.json') as f:
        data = json.load(f)

    authors = []
    texts = []
    for d in data:
        try:
            print(d['path'])
            texts.append(pdf.convert_pdf_to_txt(d['path']))
            authors.append(d['author'])
        except Exception as e:
            print('Error converting pdf', str(e))

    return authors, texts
