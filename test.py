from .util import Sentiments
from .trainingset import TrainingSet

class CVTest(object):
    def __init__(self, classifierClass, n, preprocessor, classArgs={}): 
        self.ratio = 3 # 1:2
        self.trainingSets = list()
        for i in range(self.ratio):
            self.trainingSets.append(TrainingSet(n))
        self.testingData = list()
        self.classifierClass = classifierClass
        self.n = n
        self.preprocessor = preprocessor
        self.classArgs = classArgs

    def test(self, positives, negatives):  
        results = list()
        for i in range(self.ratio):
            print("Train {}: ".format(i+1), end='')
            posTestingData = self.add(i, positives, Sentiments.POS)
            negTestingData = self.add(i, negatives, Sentiments.NEG)
            testingData = posTestingData + negTestingData
            classifier = self.getClassifer(self.trainingSets[i], self.classifierClass)
            print('OK')
            print("Test {}: ".format(i+1), end='')
            s = 0
            for filename, sentiment in testingData:
                tokens = list(line.rstrip() for line in open(filename, 'r'))
                tokens = self.preprocessor(tokens)
                result, prob = classifier.classify(tokens)
                results.append((result, sentiment))
                if result == sentiment:
                    s += 1
            print("{} ".format(s / len(testingData)), end='')
            print('OK')
        return results  
        
    def add(self, i, l, sentiment):
        testingData = list()
        trainingData = list()
        for j in range(len(l)):
            if j % self.ratio == i:
                testingData.append((l[j], sentiment))
            else:
                trainingData.append(l[j])
        self.addTrainingSet(i, sentiment, trainingData)
        return testingData
             
    def addTrainingSet(self, index, sentiment, files):
        for f in files:
            tokens = list(line.rstrip() for line in open(f, 'r'))
            tokens = self.preprocessor(tokens)
            self.trainingSets[index].add(sentiment, tokens)

    def getClassifer(self, ts, cls):
        return cls(ts, **self.classArgs)

