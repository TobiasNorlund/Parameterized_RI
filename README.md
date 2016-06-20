# The Use of Distributional Semantics in Text Classification
Word embeddings such as `word2vec`, `GloVe` and other Distributional Semantic Models have in the latest past been succesfully applied to various Natural Language Processing tasks. Many such tasks implies some kind of classification, for example sentiment analysis and text categorization.

This repository constitutes the main code of my Master's Thesis [1], where I compare the Random Indexing [2] embedding to other popular embeddings by their performance in supervised learning on sentiment analysis tasks. I also take two approaches to update the embeddings jointly with the task to boost the performance.

## Dependencies
The code is written for python 2.7 with the following dependencies:
* numpy (v1.10.1)
* theano (v0.7.0)
* lasagne (v0.1)
* scikit-learn (v0.17)

## Running the thesis experiments
1. To run the experiments in the thesis, begin by making sure you have the dependencies installed. 
2. Clone the repository:
```
git clone https://github.com/tobiasnorlund/Parameterized_RI.git
```
3. [Download](http://www.tobias.norlund.se/portfolio/masters-thesis/) or build the embeddings you intend to use. 
  * The Random Indexing vectors (RI, SGD_RI, ATT_RI) were generated using http://github.com/tobiasnorlund/CorpusParser.git
  * The Skip-Gram embeddings were generated using https://github.com/danielfrg/word2vec
  * The GloVe embeddings were generated using http://nlp.stanford.edu/projects/glove/
4. Edit `run.py` and set the paths to the embeddings
5. Run an experiment by:
```
python run.py BOW|RAND|PMI|RI|SGD_RI|ATT_RI|SG|GL MLP|CNN PL05|SST
```
Additionally, to run batches and/or replications, run:
```
python batch.py <emb(s)> <mdl(s)> <ds()> [<replications>=1]
```
for example:
```
python batch.py RI,SGD_RI - SST 2
```
runs two replications each of the RI MLP SST, RI CNN SST, SGD_RI MLP SST, SGD_RI CNN SST experiments


## Documentation
To separate concerns, the code is split up in three python packages, *embedding*, *model* and *dataset* that contains the embeddings, models and datasets used in the thesis. This goes in line with the pipeline described in the thesis, where we -using well defined interfaces- can combine any combination of embedding, model and dataset to form an experiment. See the packages respective `__init__.py` for exact interface definitions.

Conceptual experiment:
```
 -----------------       --------------------      ------------------
|                 |     |                    |    |                  |
|   Embedding     |---->|       Model        |<---|      Dataset     |
|                 |     |                    |    |                  |
 -----------------       --------------------      ------------------
```

An Embedding implementation supplies the model with a Theano expression of a n-by-d matrix, where row i correspond to the i:th word vector in the sentence to be classified.


## References

[1]   Norlund, T., *The Use of Distributional Semantics in Text Classification Models*, 2016, http://www.tobias.norlund.se/portfolio/masters-thesis

[2]   Sahlgren, M.,  *An Introduction to Random Indexing*, 2005
