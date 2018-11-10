from .util import Sentiments

class TrainingSet(object):
    def __init__(self, BagClass, grams):
        self.sentimentCount = dict((s, 0) for s in Sentiments)
        self.BagClass = BagClass
        self.grams = grams
        self.bags = list() 
        self.computed = False

    def add(self, sentiment, tokens):
        self.bags.append(self.BagClass(self.grams, tokens, sentiment))  
        self.sentimentCount[sentiment] += 1

    def getClassProb(self, sentiment):
        return self.sentimentCount[sentiment] / sum(self.sentimentCount.values())

