import numpy as np
from scipy.stats import binom
from scipy.special import comb
from ..util import Sentiments
from .classifier import Classifier

class CV(object):
    def __init__(self, fold, stemmers, gramLevels, bagClasses, classifierClasses, cutoffs):
        self.fold = fold
        self.stemmers = stemmers
        self.gramLevels = gramLevels
        self.classifierClasses = classifierClasses
        self.bagClasses = bagClasses
        self.cutoffs = cutoffs

        self.results = dict()
        self.accuracies = dict()
        self.actual = list()

        for s in stemmers:
            self.results[s] = dict()
            self.accuracies[s] = dict()
            for g in range(len(gramLevels)):
                self.results[s][g] = dict()
                self.accuracies[s][g] = dict()
                for B in bagClasses:
                    self.results[s][g][B] = dict()
                    self.accuracies[s][g][B] = dict()
                    for C in classifierClasses:
                        self.results[s][g][B][C] = dict()
                        self.accuracies[s][g][B][C] = dict()
                        for c in self.cutoffs:
                            self.results[s][g][B][C][c] = list()

    def forEach(self, func):
        for s in self.stemmers:
            for g in range(len(self.gramLevels)):
                for B in self.bagClasses:
                    for C in self.classifierClasses:
                        for c in self.cutoffs:
                            func(s, g, B, C, c)

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
            
            def getResults(s, g, B, C, c): 
                classifier = Classifier(B, C, s, self.gramLevels[g], c)
                classifier.train(trainX, trainY) 
                self.results[s][g][B][C][c].extend(classifier.test(testX))
            self.forEach(getResults)

    def exec(self, data):
        self.compute(data)
        self.getAccuracies()
        self.printAccuracies()
        signTestResults = self.signTest()

        print("Sign Test")
        for k, v in signTestResults.items():
            if hasattr(k, '__name__'): # Class
                s = k.__name__
            elif type(k) == int:
                s = self.gramLevels[k]
            else:
                s = str(k)
            print("{}\t{}".format(s, v))

    def getAccuracies(self):
        def getAccuracy(s, g, B, C, c):
            self.accuracies[s][g][B][C][c] = self.calculateAccuracy(list(zip(self.results[s][g][B][C][c], self.actual)))
        self.forEach(getAccuracy)

    def printAccuracies(self):
        def strify(s, g, B, C, c):
            accuracy = self.accuracies[s][g][B][C][c]
            print(self.stringify(s, g, B, C, c, accuracy))
        self.forEach(strify)

    def signTest(self):
        bs = self.stemmers[0]
        bg = 0
        bB = self.bagClasses[0]
        bC = self.classifierClasses[0]
        bc = self.cutoffs[0]
        baseline = self.results[bs][bg][bB][bC][bc]
        results = dict()
        
        for s in self.stemmers[1:]:
            results[s] = self.calculateSignTest(baseline, self.results[s][bg][bB][bC][bc])
        for g in range(1, len(self.gramLevels)):
            results[g] = self.calculateSignTest(baseline, self.results[bs][g][bB][bC][bc])
        for B in self.bagClasses[1:]:
            results[B] = self.calculateSignTest(baseline, self.results[bs][bg][B][bC][bc])
        for C in self.classifierClasses[1:]:
            results[C] = self.calculateSignTest(baseline, self.results[bs][bg][bB][C][bc])
        for c in self.cutoffs[1:]:
            results[c] = self.calculateSignTest(baseline, self.results[bs][bg][bB][bC][c])
        
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
        return 2 * binom.cdf(k, N, q)

    def stringify(self, s, g, B, C, c, x):
        return "{}\t{}\t{}\t{}\t{}\t{}".format(s.__name__, self.gramLevels[g], B.__name__, C.__name__, c, x)

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

