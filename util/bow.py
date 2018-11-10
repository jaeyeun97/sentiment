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
    def __init__(self, n):
        self.n = n
        self.tokenMap = dict()
        self.sentimentCount = dict([(s, 0) for s in Sentiments])

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

    def getTokenCount(self, token, sentiment):
        if token in self.tokenMap:
            return self.tokenMap[token][sentiment]
        else:
            return 1

class BagOfPresence(object):
    def __init__(self, n):
        self.n = n
        self.tokenMap = dict((s, set()) for s in Sentiments)

    def getSentimentCount(self, sentiment):
        return len(self.tokenMap[sentiment])

    def addToken(self, sentiment, token):
        self.tokenMap[sentiment].add(token)

    def addTokens(self, sentiment, tokens): 
        for token in getNgramTokens(self.n, tokens):
            self.addToken(sentiment, token) 

    def getTokenCount(self, token, sentiment):
        if token in self.tokenMap[sentiment]:
            return 1
        else:
            return 0


