# Intro

This repository contains an implementation of Naive Bayes classifier, utilization of `scikit-learn`'s SVM implementation and multiple bag-of-words based feature extractions for replicating the results of Pang et al. 2002, for the NLP course at Cambridge.

I do realise this code is highly inefficient and has fairly straightforward changes that may improve the speed of the classifications significantly, but I had no problem just running it overnight while I sleep (and the purpose of this code is to get the results so whatever).

If you want to tweak the number of classifications to run, just change the `__main__.py`. It will iterate through everything by default.

# Run

At the root directory of the repository:
```
$ pip install -r requirements.txt
$ python -m sentiment
```
