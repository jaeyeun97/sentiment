import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction import DictVectorizer
from ..util import Sentiments

class SVM(object):
    def __init__(self, trainingSet):
        self.trainingSet = trainingSet
        self.rlookup = dict()

        features = list()
        classes = list()
        for bag in trainingSet.bags:
            features.append(dict((t, bag.getTokenCount(t)) for t in bag.tokenMap))
            classes.append(bag.sentiment.value)

        self.vectorizer = DictVectorizer(sparse=True, dtype=int)
        X = self.vectorizer.fit_transform(features)
        y = np.array(classes)

        self.classifier = SVC(kernel='linear', gamma='auto')
        self.classifier.fit(X, y)

    def classify(self, tokens):
        tokenMap = self.trainingSet.BagClass.generateTokenMap(self.trainingSet.grams, tokens)
        X = self.vectorizer.transform(tokenMap) 
        y = self.classifier.predict(X)
        return Sentiments(y[0])
