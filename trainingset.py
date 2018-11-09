from .util import Sentiments, BagOfWords, Gram

class TrainingSet(object):
    def __init__(self, n):
        self.sentimentCount = dict((s, 0) for s in Sentiments)
        self.n = n
        self.bows = list()
        self.computed = False

        for i in range(n):
            self.bows.append(BagOfWords(i+1))

        self.grams = [Gram(self.bows[0])]
        for i in range(1, self.n):
            self.grams.append(Gram(self.bows[i], self.bows[i-1]))

    def add(self, sentiment, tokens):
        # tokens = list(line.rstrip() for line in open(filename, 'r'))
        for bow in self.bows:
            bow.addTokens(sentiment, tokens)
        self.sentimentCount[sentiment] += 1

    def getClassProb(self, sentiment):
        return self.sentimentCount[sentiment] / sum(self.sentimentCount.values())

