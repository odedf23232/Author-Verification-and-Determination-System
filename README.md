### Resources

- [Text Processing SciKit](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- [ML Tutorials Point](https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_classification_algorithms_logistic_regression.htm)
- [ML for Author Identification](https://towardsdatascience.com/a-machine-learning-approach-to-author-identification-of-horror-novels-from-text-snippets-3f1ef5dba634)
- [Vector Space Model](https://towardsdatascience.com/lets-understand-the-vector-space-model-in-machine-learning-by-modelling-cars-b60a8df6684f)
- [Vector Space Model Impl](https://github.com/peermohtaram/Vector-Space-Model/blob/master/Vector_Space_Model.ipynb)
- [Vector Space Model2](https://www.johnwittenauer.net/language-exploration-using-vector-space-models/)
- [Bag of Words](https://www.freecodecamp.org/news/an-introduction-to-bag-of-words-and-how-to-code-it-in-python-for-nlp-282e87a9da04/)
- [TfidfVectorizer vs Transformer](https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/#Tfidftransformer-vs-Tfidfvectorizer)
- [Extract Keywords](https://www.freecodecamp.org/news/how-to-extract-keywords-from-text-with-tf-idf-and-pythons-scikit-learn-b2a0f3d7e667/)


### Datasets

`datasets.py` contains functions for supplying datasets. Each provides a list of authors
and a list of raw text from the documents. The are the following datasets:
- local_books: dataset of books in PDF format which are stored on the file system
- web_articles: dataset of news articles from the web 

# Analytics

- Match between CHUNK_SIZE and results
- Match between distance functions and results
    - implement all distance functions
    
# Requirements

- `numpy` - for scientific computing in Python. Provides some high-performance objects. Used by the other libraries 
behind the scenes mostly.
- `pdfminer` - working with PDF format. Using in `pdf.py` for converting PDF into text.
- `pandas` - data analysis tool, used to analyze results and information in `analysis.py`
- `sklearn`/`scikit-learn` - machine learning in python. Provides some algorithms for the transformation
- `scipy` - ecosystem for math, science and engineering. Provides distance algorithms, and some math structures