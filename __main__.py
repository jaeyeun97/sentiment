import glob
from .classifier import NaiveBayes, SVM
from .util import Sentiments, BagOfWords, BagOfPresence
from .test import CVTest
from .stemmer import PorterStemmer

base_dir = './data/tokenized'
data = dict()
for s in Sentiments:
    data[s] = [f for f in glob.glob("{}/{}/*.tag".format(base_dir, s.name))]
    data[s].sort()
    
p = PorterStemmer()

stemmer = lambda tokens: list(p.stem(word, 0, len(word)-1).lower() for word in tokens)
nonstemmer = lambda tokens: tokens

test = CVTest(3, BagOfPresence, NaiveBayes, stemmer, {1, 2})
results = test.test(data)

s = 0
for test, actual in results:
    if test == actual: 
        s += 1
print(s / len(results))

