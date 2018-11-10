import numpy as np
import glob
from ..util import Sentiments, getNgramTokens

class NaiveBayes(object):
    """
    Take an input of a set of n-gram models to calculate probability
    """

    def __init__(self, trainingSet):
        self.trainingSet = trainingSet

    def calculate(self, tokens): 
        prob = dict((s, np.log(self.trainingSet.getClassProb(s))) for s in Sentiments)

        for s in Sentiments:
            total = np.log(sum(bow.getSentimentCount(s) for bow in self.trainingSet.bows))
            for bow in self.trainingSet.bows:
                ts = getNgramTokens(bow.n, tokens)
                for t in ts:
                    prob[s] += np.log(bow.getTokenCount(t,s)) - total
        return prob
        
    def classify(self, tokens):
        probs = self.calculate(tokens)
        return max(probs.keys(), key=lambda key: probs[key])
