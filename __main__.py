import glob
from .classifier import NaiveBayes, SVM
from .util import Sentiments, BagOfWords, BagOfPresence
from .test import CV
from .stemmer import PorterStemmer

base_dir = './data/tokenized'
data = dict()
for s in Sentiments:
    data[s] = [f for f in glob.glob("{}/{}/*.tag".format(base_dir, s.name))]
    data[s].sort()
    
p = PorterStemmer()

def porter(tokens):
    return list(p.stem(word, 0, len(word)-1).lower() for word in tokens)

def nonstemmer(tokens):
    return tokens

stemmers = [nonstemmer, porter]
gramLevels = [{1}, {1,2}, {2}]
bagClasses = [BagOfWords, BagOfPresence]
classifierClasses = [NaiveBayes, SVM]

cv = CV(3, stemmers, gramLevels, bagClasses, classifierClasses)
cv.exec(data)
