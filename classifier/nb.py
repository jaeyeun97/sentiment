import numpy as np
import glob
from ..util import Sentiments, getNgramTokens

class NaiveBayes(object):
    def __init__(self, trainingSet):
        self.trainingSet = trainingSet 
        self.count = dict((s, 0) for s in Sentiments) 
        self.tokenMap = dict((s, dict()) for s in Sentiments)
        self.grams = set()

        for bag in trainingSet.bags:
            self.grams.update(bag.grams)
            for t in bag.tokenMap:
                if t not in self.tokenMap[bag.sentiment]:
                    self.tokenMap[bag.sentiment][t] = 0
                self.tokenMap[bag.sentiment][t] = bag.add(self.tokenMap[bag.sentiment][t], t)
    
        for s, v in self.tokenMap.items():
            for t, n in v.items():
                self.count[s] += n

    def getTokenCount(self, sentiment, token):
        if token in self.tokenMap[sentiment]:
            return self.tokenMap[sentiment][token]
        else:
            return 0

    def calculate(self, tokens): 
        prob = dict((s, np.log(self.trainingSet.getClassProb(s))) for s in Sentiments)
        for i in self.grams:
            ts = getNgramTokens(i, tokens)
            for s in Sentiments:
                count = np.log(self.count[s] + len(ts))
                for t in ts:
                    prob[s] += np.log(self.getTokenCount(s, t) + 1) - count
        return prob
        
    def classify(self, tokens):
        probs = self.calculate(tokens)
        return max(probs.keys(), key=lambda key: probs[key])
