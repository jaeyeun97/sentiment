import numpy as np
import glob
from ..util import Sentiments, getNgramTokens

class NaiveBayes(object):
    """
    Take an input of a set of n-gram models to calculate probability
    """

    def __init__(self, trainingSet, l=0.8):
        self.trainingSet = trainingSet
        self.grams = trainingSet.grams 
        self.n = trainingSet.n
        self.l = l

    def calculate(self, tokens): 
        tokens = getNgramTokens(self.n, tokens)
        probSum = dict((s, np.log(self.trainingSet.getClassProb(s))) for s in Sentiments)

        for token in tokens:
            # Probability of being a certain sentiment in the first place
            for s in Sentiments:
                prob = self.grams[0].getProb(token[-1], s)
                for i in range(self.n): 
                    t = token[self.n - i - 1:]
                    prob = prob * (1-self.l) + self.l * self.grams[i].getProb(t, s)
                probSum[s] += np.log(prob)
        return probSum
        
    def classify(self, tokens):
        probs = self.calculate(tokens)
        return max(probs.keys(), key=lambda key: probs[key]), probs
