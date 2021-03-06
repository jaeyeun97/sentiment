import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction import DictVectorizer
from ..util import Sentiments

class SVM(object):
    def __init__(self, trainingSet):
        self.trainingSet = trainingSet
        self.rlookup = dict()

        X_data = list()
        y_data = list()
        features = trainingSet.getFeatures()

        for bag in trainingSet.bags:
            X_data.append({t: bag.getTokenCount(t) for t in bag.tokenMap})
            y_data.append(bag.sentiment.value)

        self.vectorizer = DictVectorizer(sparse=True, dtype=int)
        X = self.vectorizer.fit_transform(X_data)
        y = np.array(y_data)

        self.classifier = SVC(kernel='linear', gamma='auto')
        self.classifier.fit(X, y)

    def classify(self, tokensList):
        tokenMaps = list(self.trainingSet.BagClass.generateTokenMap(self.trainingSet.grams, ts) for ts in tokensList)
        X = self.vectorizer.transform(tokenMaps)
        ys = self.classifier.predict(X)
        return [Sentiments(y) for y in ys]

    def __str__(self):
        return "SVM"
