---
title: NLP Practical 1 - Sentiment Classification Revisited
author: Charles J.Y. Yoon
institute: University of Cambridge
date: 11^th^ November, 2018
classoption: twocolumn
documentclass: article
header-includes:
	- \usepackage{multirow}
	- \usepackage{amsmath}
	- \DeclareMathOperator*{\argmax}{arg\!\max}
---

\begin{figure*}
\begin{center}
\begin{tabular}{lr|c|c|c|c|}
\cline{3-6}
 & \multicolumn{1}{l|}{} & \multicolumn{2}{c|}{Naive Bayes} & \multicolumn{2}{c|}{SVM} \\ \cline{3-6} 
 & \multicolumn{1}{l|}{} & \multicolumn{1}{l|}{Frequency} & \multicolumn{1}{l|}{Presence} & \multicolumn{1}{l|}{Frequency} & \multicolumn{1}{l|}{Presence} \\ \hline
\multicolumn{1}{|r|}{\multirow{2}{*}{Unigram}} & Stemmed & 79.95 & 81.45 & 83.10 & 83.75 \\ \cline{2-6} 
\multicolumn{1}{|r|}{} & Unstemmed & 79.85 & 81.55 & 82.95 & 84.95 \\ \hline
\multicolumn{1}{|l|}{\multirow{2}{*}{Unigram + Bigram}} & Stemmed & 79.70 & 79.20 & 84.40 & 86.55 \\ \cline{2-6} 
\multicolumn{1}{|l|}{} & Unstemmed & 79.20 & 76.95 & 83.65 & 86.55 \\ \hline
\multicolumn{1}{|r|}{\multirow{2}{*}{Bigram}} & Stemmed & 77.15 & 75.85 & 80.60 & 82.30 \\ \cline{2-6} 
\multicolumn{1}{|r|}{} & Unstemmed & 74.10 & 72.00 & 79.90 & 81.10 \\ \hline
\end{tabular}
\end{center}
\caption{Accuracies in percentage given configuration} \label{fig:acc}
\end{figure*}

# Introduction

For the first practical, document sentiment classification was revisited. Pang et al. describes multiple methods for sentiment classification, specifically of movie reviews. This domain is experimentally convenient, since there is a large dataset with classification, taken by the rating reviewers assign. We have been given a similar dataset that has been tokenized for classification, in the context of the NLP course.

The original experiment used a number of supervised learning methods with multiple models of feature extraction, but comes short in explaining the effectiveness of different models. We aim to take a more prudent approach in choosing our language model, and evaluate the effectiveness of each sub-model we choose. We will, however, exclude Maximum Entropy (ME) model that has been used in the original paper.

# Methods

## Classifiers

We will be using Naive Bayes and Select Vector Machine classifiers on binary classes.

### Naive Bayes

For a given feature vector $\vec{d}$ for a document, Naive Bayes classifiers uses Bayes rule to compute:

\begin{align}
	& \argmax_{c \in C} P(c | \vec{d}) \nonumber\\
	&= \argmax_{c \in C} P(\vec{d}| c) P(c) \nonumber \\
	&= \argmax_{c \in C} \prod_{i = 0}^{n} P(d_i | c) P(c)  \label{nb:prod} \\
	&= \argmax_{c \in C} \log P(c) + \sum_{i = 0}^{n} \log P(d_i | c) \label{nb:sum} 	
\end{align}

Where $d_i$ are each features of the document. Since the computation for equation \ref{nb:prod} may underflow, we will take the log as shown in equation \ref{nb:sum}.

### SVM

SVM classifiers on the other hand, compute the equation:

\begin{equation}
	\vec{w} = \sum_{i} \alpha_ic_i\vec{d}_i \nonumber
\end{equation}

for $c_i \in {-1, 1}$ is the classification, and $\vec{d}_i$ are the feature vector extracted from the document. $\alpha_i$ are calculated by an optimization problem to select vectors that form $\vec{w}$, which in this case will be handled by the implementation at `scikit-learn.svm.SVC`.

## Bag of Words

Each document is represented as a vector by a bag-of-words model, an accumulation of either frequency or occurrences of words seen in the training set. Any $n$-gram model can be used to create the feature vectors--here we have used unigram and bigrams as per the original.

### Frequency vs Presence

By testing on both types of bag-of-words models, we aim to see the difference in accuracy between frequency and presence.

### $n$-gram feature extraction

The unigram and bigram models used in sentiment classification is different to that of language generation. While such models may calculate the bigram probability as $P(x_i | x_{i-1})$, here we *take each unigram and bigram as a feature*, since we are not interested in the generative probability of words.

### Stemming

Unlike the original experiment, we will be using a stemmer to preprocess the document, specifically the Porter Stemmer. Since the meaning of the word do not change with the change in morphology, stemming the words may increase the accuracy of the classifier.

# Evaluaton and Results

We have performed two different tests, accuracy comparison and sign test, in order to determine the effectiveness of each change that we have made to the language model. A stratified 3-fold cross-validation set by round-robin, and the predictions were concatenated to be evaluated at once.

## Accuracy Comparison

Figure \ref{fig:acc} shows accuracies in percentage values given different configurations of feature extraction and classification models. In order to verify that the change made indeed makes a difference, sign tests have been performed.

## Sign Tests

\begin{figure}
\begin{center}
\begin{tabular}{rc}
\hline
\multicolumn{1}{l}{Change in model} & \multicolumn{1}{l}{p-value} \\ \hline
Stemming & 0.982 \\
Unigram+Bigram & 0.789 \\
Bigram & 0.011 \\
SVM & 0.461 \\
Occurances & 0.173 \\ \hline
\end{tabular}
\end{center}
\caption{Sign test results between baseline and individual change} \label{fig:sign}
\end{figure}

We assumed that each change are independent and chose a baseline classifier (Naive Bayes, no stemming, unigrams, and frequency) to be compared pairwise to a set of models that has only one element changed. \ref{fig:sign} shows the `p-values` of each comparison, where higher `p-value` notes less significance in change (in favor of the null hypothesis).

# Conclusion

From the results of figures \ref{fig:acc} and \ref{fig:sign}, we can conclude that SVM classifier has a significant improvement, the accuracies rise and the null hypothesis seems defeated. Using occurance instead of frequency seems to yield different results, but the accuracies do not favor one or the other; further experiments can see whether one of these models favor a certain classifier but is out of the scope of this report. Stemming, on the other hand, did not make a significant difference in the model, unlike our previous predictions.

The significant difference in the unigram+bigram and bigram model was anticipated as the feature vector yielded differs; bigram models have performed worse than unigram models, however, which can be the lack of each token in data affecting the results.


