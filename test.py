from .util import Sentiments
from .trainingset import TrainingSet

class CVTest(object):
    def __init__(self, fold, BagClass, ClassifierClass, preprocessor, grams={1}): 
        self.fold = 3 
        self.BagClass = BagClass
        self.ClassifierClass = ClassifierClass
        self.preprocessor = preprocessor
        self.grams = grams

    def test(self, data):  
        results = list()
        for i in range(self.fold):
            print("Train {}: ".format(i+1), end='')
            trainingSet = TrainingSet(self.BagClass, self.grams)
            testingData = list()
            
            for s in Sentiments:
                trainingData, testData = self.split(i, data[s], s)
                self.addFiles(trainingSet, s, trainingData)
                testingData += testData

            classifier = self.getClassifer(trainingSet)

            print('OK')

            print("Test {}: ".format(i+1), end='')
            s = 0
            for filename, sentiment in testingData:
                tokens = list(line.rstrip() for line in open(filename, 'r'))
                tokens = self.preprocessor(tokens)
                result = classifier.classify(tokens)
                results.append((result, sentiment))
                if result == sentiment:
                    s += 1
            print("{} ".format(s / len(testingData)), end='')
            print('OK')
        return results

    def split(self, index, l, sentiment):
        testingData = list()
        trainingData = list()
        for j in range(len(l)):
            if j % self.fold == index:
                testingData.append((l[j], sentiment))
            else:
                trainingData.append(l[j])
        return trainingData, testingData
             
    def addFiles(self, trainingSet, sentiment, files):
        for f in files:
            tokens = list(line.rstrip() for line in open(f, 'r'))
            tokens = self.preprocessor(tokens)
            trainingSet.add(sentiment, tokens)

    def getClassifer(self, ts):
        return self.ClassifierClass(ts) 
