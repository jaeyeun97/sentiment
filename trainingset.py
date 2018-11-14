from .util import Sentiments

class TrainingSet(object):
    def __init__(self, BagClass, grams, cutoff):
        self.sentimentCount = dict((s, 0) for s in Sentiments)
        self.BagClass = BagClass
        self.grams = grams
        self.cutoff = cutoff
        self.bags = list() 
        self.features = None
        self.totalTokenCount = dict()

    def add(self, sentiment, tokens):
        bag = self.BagClass(self.grams, tokens, sentiment)
        for t in bag.tokenMap:
            if t not in self.totalTokenCount:
                self.totalTokenCount[t] = 0
            self.totalTokenCount[t] += bag.getTokenCount(t)
        self.bags.append(bag)    
        self.sentimentCount[sentiment] += 1

    def getFeatures(self):
        if self.features is None:
            self.features = set(t for t, v in self.totalTokenCount.items() if v >= self.cutoff)
        return self.features

    def getClassProb(self, sentiment):
        return self.sentimentCount[sentiment] / sum(self.sentimentCount.values())

    def __str__(self):
        return "{} {}".format(self.BagClass.__name__, self.grams)
