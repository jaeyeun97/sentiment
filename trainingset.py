from .util import Sentiments, BagOfWords

class TrainingSet(object):
    def __init__(self, grams):
        self.sentimentCount = dict((s, 0) for s in Sentiments)
        self.bows = list()
        self.grams = grams
        self.computed = False

        for n in grams:
            self.bows.append(BagOfWords(n))

    def add(self, sentiment, tokens):
        for bow in self.bows:
            bow.addTokens(sentiment, tokens)
        self.sentimentCount[sentiment] += 1

    def getClassProb(self, sentiment):
        return self.sentimentCount[sentiment] / sum(self.sentimentCount.values())

