from ..util import Sentiments
from ..trainingset import TrainingSet

class Classifier(object):
    def __init__(self, BagClass, ClassifierClass, preprocessor, grams={1}): 
        self.BagClass = BagClass
        self.ClassifierClass = ClassifierClass
        self.preprocessor = preprocessor
        self.grams = grams
        self.classifier = None

    def train(self, trainX, trainY):  
        assert len(trainX) == len(trainY)
        trainingSet = TrainingSet(self.BagClass, self.grams)
        for i in range(len(trainX)):
            self.addFile(trainingSet, trainY[i], trainX[i])
        self.classifier = self.getClassifer(trainingSet)

    def test(self, files):
        tokens = [self.preprocessor(list(l.rstrip() for l in open(f, 'r'))) for f in files]
        return self.classifier.classify(tokens)

    def addFile(self, trainingSet, sentiment, filename):
        tokens = list(line.rstrip() for line in open(filename, 'r'))
        tokens = self.preprocessor(tokens)
        trainingSet.add(sentiment, tokens)

    def getClassifer(self, ts):
        return self.ClassifierClass(ts) 
