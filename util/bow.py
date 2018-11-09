import numpy as np
from .sentiments import Sentiments

def getNgramTokens(n, tokens):
    l = list()
    for i in range(n):
        index = -1 * n + i + 1
        if index == 0:
            tl = tokens[i:]
        else:
            tl = tokens[i:index]
        l.append(tl)
    return list(zip(*l))


class BagOfWords(object):
    """
    n-gram bag of words.
    self.probs contains the joint probability P(x_i, ... , x_i-n | c)
    """

    def __init__(self, n):
        self.n = n
        self.tokenMap = dict()
        self.sentimentCount = dict([(s, 0) for s in Sentiments])
        self.prob = None

    def getSentimentCount(self, sentiment):
        return self.sentimentCount[sentiment]

    def addToken(self, sentiment, token):
        if token not in self.tokenMap:
            self.tokenMap[token] = {Sentiments.POS: 1, Sentiments.NEG: 1}
        self.tokenMap[token][sentiment] += 1
        self.sentimentCount[sentiment] += 1

    def addTokens(self, sentiment, tokens): 
        for token in getNgramTokens(self.n, tokens):
            self.addToken(sentiment, token) 

    def getTokenMap(self):
        return self.tokenMap

    def getTokenCount(self, token, sentiment):
        if token in self.tokenMap:
            return self.tokenMap[token][sentiment]
        else:
            return 1

#     def getProb(self): 
#         if not self.prob:
#             self.prob = dict((k, dict((s, v[s] / self.getSentimentCount(s)) for s in Sentiments)) for k, v in self.tokenMap.items())
#         return self.prob
# 
#     def defaultProb(self, s):
#         return 1 / self.getSentimentCount(s)
