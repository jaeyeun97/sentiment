import glob
from .classifier import NB, SVM
from .util import Sentiments, BagOfFrequency, BagOfPresence
from .test import CV, Classifier
from .stemmer import PorterStemmer

base_dir = './data/tokenized'
data = dict()
for s in Sentiments:
    data[s] = [f for f in glob.glob("{}/{}/*.tag".format(base_dir, s.name))]
    data[s].sort()
 
p = PorterStemmer()

def porter(tokens):
    return list(p.stem(word, 0, len(word)-1).lower() for word in tokens)

def nostem(tokens):
    return tokens

stemmers = [nostem, porter]
gramLevels = [{1}, {1,2}, {2}]
bagClasses = [BagOfFrequency, BagOfPresence]
classifierClasses = [NB, SVM]
cutoffs = [0, 3]

cv = CV(3, stemmers, gramLevels, bagClasses, classifierClasses, cutoffs)
cv.exec(data)

# trainX = list()
# trainY = list()
# testX = list()
# testY = list()
# for s in Sentiments:
#     trainX.extend(data[s][:666])
#     testX.extend(data[s][666:])
#     trainY.extend([s]*666)
#     testY.extend([s]*334)
# classifier = Classifier(BagOfFrequency, SVM, porter, {1,2}, 0)
# classifier.train(trainX, trainY)
# classifier.test(testX)
