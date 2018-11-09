import glob
from .classifier import NaiveBayes
from .util import Sentiments, getNgramTokens
from .trainingset import TrainingSet
from .test import CVTest

base_dir = './data/tokenized'
positives = [f for f in glob.glob("{}/{}/*.tag".format(base_dir, 'POS'))]
negatives = [f for f in glob.glob("{}/{}/*.tag".format(base_dir, 'NEG'))]

positives.sort()
negatives.sort()


# test = CVTest(NaiveBayes, 2, lambda x: x, {'l': 0.8})
test = CVTest(NaiveBayes, 2, lambda x: x, {'l': 0.7})
results = test.test(positives, negatives)

s = 0
for test, actual in results:
    if test == actual: 
        s += 1
print(s / len(results))

