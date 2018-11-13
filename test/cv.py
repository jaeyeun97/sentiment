from ..util import Sentiments
from .classifier import Classifier
from .sign import SignTest

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
                print("Train:\t{}".format(self.stringify(s, g, B, C, c))) 
                classifier = Classifier(B, C, s, self.gramLevels[g], c)
                classifier.train(trainX, trainY) 
                print("Test:\t{}".format(self.stringify(s, g, B, C, c))) 
                self.results[s][g][B][C][c].extend(classifier.test(testX))
            self.forEach(getResults)

    def exec(self, data):
        self.compute(data)
        self.accuracyTest()
        self.signTest()

    def accuracyTest(self):
        print("Accuracy Test")
        def printAccuracy(s, g, B, C, c):
            accuracy = self.calculateAccuracy(list(zip(self.results[s][g][B][C][c], self.actual)))
            print("{}:\t{}".format(self.stringify(s, g, B, C, c), accuracy))
        self.forEach(printAccuracy)

    def signTest(self):
        print("Sign Test")
        test = SignTest(self.actual) 
        def getFirst(s1, g1, B1, C1, c1):
            firstResult = self.result[s1][g1][B1][C1][c1]
            def getSecond(s2, g2, B2, C2, c2):
                secondResult = self.result[s2][g2][B2][C2][c2]
                result = test.test(firstResult, secondResult)
                print("{}\t|\t{}:\t{}".format(self.stringify(s1, g1, B1, C1, c1), self.stringify(s2, g2, B2, C2, c2), result))
            self.forEach(getSecond)
 
    def stringify(self, s, g, B, C, c):
        return "{}\t{}\t{}\t{}\t{}".format(s.__name__, self.gramLevels[g], B.__name__, C.__name__, c)

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

