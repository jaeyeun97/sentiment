import numpy as np
from scipy.stats import binom
from scipy.special import comb
from ..util import Sentiments
from .classifier import Classifier

class CV(object):
    def __init__(self, fold, stemmers, gramLevels, bagClasses, classifierClasses):
        self.fold = fold
        self.stemmers = stemmers
        self.gramLevels = gramLevels
        self.classifierClasses = classifierClasses
        self.bagClasses = bagClasses

        self.classifiers = dict()
        self.results = dict()
        self.accuracies = dict()
        self.actual = list()

        for s in stemmers:
            self.classifiers[s] = dict()
            self.results[s] = dict()
            self.accuracies[s] = dict()
            for g in range(len(gramLevels)):
                self.classifiers[s][g] = dict()
                self.results[s][g] = dict()
                self.accuracies[s][g] = dict()
                for B in bagClasses:
                    self.classifiers[s][g][B] = dict()
                    self.results[s][g][B] = dict()
                    self.accuracies[s][g][B] = dict()
                    for C in classifierClasses:
                        self.results[s][g][B][C] = list()
                        self.classifiers[s][g][B][C] = Classifier(B, C, s, gramLevels[g])

    def forEach(self, func):
        for s in self.stemmers:
            for g in range(len(self.gramLevels)):
                for B in self.bagClasses:
                    for C in self.classifierClasses:
                        func(s, g, B, C)

    def compute(self, data):
        for i in range(self.fold):
            trainX = list()
            trainY = list()
            testX = list()
            testY = list()
            for s in Sentiments:
                X1, X2 = self.split(self.fold, i, data[s])
                trainX.extend(X1)
                trainY.extend([s]*len(X1))
                testX.extend(X2)
                testY.extend([s]*len(X2))
            self.actual.extend(testY)               
            
            def getResults(s, g, B, C): 
                classifier = self.classifiers[s][g][B][C]
                classifier.train(trainX, trainY) 
                self.results[s][g][B][C].extend(classifier.test(testX))
            self.forEach(getResults)

    def exec(self, data):
        self.compute(data)
        self.getAccuracies()
        self.printAccuracies()
        signTestResults = self.signTest()

        for k, v in signTestResults.items():
            print("{} {}".format(str(k), v))

    def getAccuracies(self):
        def getAccuracy(s, g, B, C):
            self.accuracies[s][g][B][C] = self.calculateAccuracy(list(zip(self.results[s][g][B][C], self.actual)))
        self.forEach(getAccuracy)

    def printAccuracies(self):
        def strify(s, g, B, C):
            accuracy = self.accuracies[s][g][B][C]
            print(self.stringify(s, g, B, C, accuracy))
        self.forEach(strify)

    def signTest(self):
        bs = self.stemmers[0]
        bg = 0
        bbc = self.bagClasses[0]
        bcc = self.classifierClasses[0]
        baseline = self.results[bs][bg][bbc][bcc]
        results = dict()
        
        for s in self.stemmers[1:]:
            results[s] = self.calculateSignTest(baseline, self.results[s][bg][bbc][bcc])
        for g in range(1, len(self.gramLevels)):
            results[g] = self.calculateSignTest(baseline, self.results[bs][g][bbc][bcc])
        for bc in self.bagClasses[1:]:
            results[bc] = self.calculateSignTest(baseline, self.results[bs][bg][bc][bcc])
        for cc in self.classifierClasses[1:]:
            results[cc] = self.calculateSignTest(baseline, self.results[bs][bg][bbc][cc])
        
        return results

    def calculateSignTest(self, baseline, target):
        assert len(baseline) == len(target)
        null = 0
        plus = 0
        minus = 0
        for i in range(len(baseline)):
            if baseline[i] == target[i]:
                null += 1
            elif baseline[i] == self.actual[i]:
                minus += 1
            elif target[i] == self.actual[i]:
                plus += 1

        N = 2 * np.ceil(null/2) + plus + minus
        k = np.ceil(null/2) + np.minimum(plus, minus)
        q = 0.5
        # return np.prod([2, np.sum(np.prod([comb(N, i, exact=True), np.power(q, i), np.power(1-q, N - i)]) for i in np.arange(k+1))])
        return 2 * binom.cdf(k, N, q)

    def stringify(self, s, g, B, C, x):
        return "{} {} {} {} {} ".format(str(s), str(self.gramLevels[g]), str(B), str(C), x)

    @staticmethod
    def split(fold, index, data):
            testingData = list()
            trainingData = list()
            for j in range(len(data)):
                if j % fold == index:
                    testingData.append(data[j])
                else:
                    trainingData.append(data[j])
            return trainingData, testingData

    @staticmethod
    def calculateAccuracy(results):
        s = 0
        for test, actual in results:
            if test == actual: 
                s += 1
        return s / len(results)

