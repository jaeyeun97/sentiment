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
    def __init__(self, grams, tokens, sentiment):
        self.grams = grams
        self.sentiment = sentiment
        self.tokenMap = self.generateTokenMap(grams, tokens) 

    @staticmethod
    def generateTokenMap(grams, tokens):
        tokenMap = dict()
        for i in grams:
            for token in getNgramTokens(i, tokens):
                if token not in tokenMap:
                    tokenMap[token] = 0
                tokenMap[token] += 1
        return tokenMap

    def add(self, existingData, token):
        return existingData + self.getTokenCount(token)

    def getTokenCount(self, t):
        return self.tokenMap[t]


class BagOfPresence(object):
    def __init__(self, grams, tokens, sentiment):
        self.grams = grams
        self.sentiment = sentiment
        self.tokenMap = self.generateTokenMap(gram, tokens) 

    @staticmethod
    def generateTokenMap(grams, tokens):
        tokenMap = set()
        for i in grams:
            for token in getNgramTokens(i, tokens):
                tokenMap.add(token)
        return tokenMap

    def add(self, existingData, token):
        return existingData | self.getTokenCount(token)

    def getTokenCount(self, t):
        return 1 if t in self.tokenMap else 0
