# Word Embeddings applied to Text Classification
Word embeddings such as `word2vec`, `GloVe` and other Distributional Semantic Models have in the latest past been succesfully applied to various Natural Language Processing tasks. Many such tasks implies some kind of classification, for example sentiment analysis and text categorization.

This repository constitutes the main code of my Master's Thesis [1], where I compare the Random Indexing [2] embedding to other popular embeddings using different models and datasets. I also try to modify the embeddings to boost the performance.

## Dependencies
The code is written for python 2.7 with the following dependencies
* numpy
* theano
* lasagne

## Running the thesis experiments
To be written...

## Documentation
To separate concerns, the code is split up in three python packages, *embedding*, *model* and *dataset* that contains the embeddings, models and datasets used in the thesis. This goes in line with the pipeline described in the thesis, where we using well defined interfaces can combine any combination of embedding, model and dataset to form an experiment.
```
 -----------------       --------------------      ------------------
|                 |     |                    |    |                  |
|   Embedding     |---->|       Model        |--->|      Dataset     |
|                 |     |                    |    |                  |
 -----------------       --------------------      ------------------
```

## References

[1]   To be announched ... 

[2]   Sahlgren, M.,  *An Introduction to Random Indexing*, 2005
