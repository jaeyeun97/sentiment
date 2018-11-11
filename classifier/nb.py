import numpy as np
import glob
from ..util import Sentiments, getNgramTokens

class NaiveBayes(object):
    def __init__(self, trainingSet):
        self.trainingSet = trainingSet 
        self.count = dict((s, 0) for s in Sentiments) 
        self.tokenMap = dict((s, dict()) for s in Sentiments)
        self.grams = trainingSet.grams

        for bag in trainingSet.bags:
            for t in bag.tokenMap:
                if t not in self.tokenMap[bag.sentiment]:
                    self.tokenMap[bag.sentiment][t] = 0
                self.tokenMap[bag.sentiment][t] += bag.getTokenCount(t)
    
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
        evalMap = self.trainingSet.BagClass.generateTokenMap(self.grams, tokens)

        for s in Sentiments:
            count = np.log(self.count[s] + sum(evalMap.values()))
            for t, cnt in evalMap.items():
                prob[s] += cnt * (np.log(self.getTokenCount(s, t) + 1) - count)
        return prob
        
    def classify(self, tokenslist):
        results = list()
        for tokens in tokenslist:
            probs = self.calculate(tokens)
            results.append(max(probs.keys(), key=lambda key: probs[key]))
        return results

    def __str__(self):
        return "Naive Bayes"
