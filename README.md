# KeywordSearchEngine

Implementation of a keyword search engine for news articles. Based on the tutorial available at https://www.kaggle.com/amitkumarjaiswal/nlp-search-engine.

The solution proposed assumes it should be strictly self contained. As such the dataset pre-processing is included in the function (it can however be substituted by loading a save file). The pre-processing consists in removing punctuaton, dividing the document in words, removing irrelevant known words and then de-capitalizing and stemming the remaining words.

After preprocessing the inverse-index is generated. The inverse-index contains for each known word in which articles and at which positions the word is found, besides this it also contains the TD-IDF / BM25 rank.

Besides this value new scores that reflect if all words of the query are present in the article and if they are close to each other are generated.

Afterwards the results are ranked based on the previous scores and a set of predefined rules.

The solution proposed is not suited for the problem at hand due to the lack of scalability of the inverse-index generation. The dataset of 50.000 articles is too large for the computation to be perform in acceptable time.

## Dataset

The dataset used is made available by kaggle at (https://www.kaggle.com/snapcrack/all-the-news/version/4#articles1.csv). However the focus will be only on the firste 50k articles available in file articles1.csv

## Future work

Substitute rule based ranking by a ML approach

Improve generation of the inverse-index (it is the most computationally expensive part and should be run only once)

Improve metrics e.g. implement full sequential query bonus

Implement multi-field search

## Author

Diogo Caldas de Oliveira
