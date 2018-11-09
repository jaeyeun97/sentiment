from .sentiments import Sentiments

class Gram(object):
    """
        contains the conditional probability P(x_i | x_i-1, ... , x_i-n, c)
    """

    def __init__(self, thisBoW, prevBoW=None):
        self.n = thisBoW.n
        self.thisBoW = thisBoW
        self.prevBoW = prevBoW

    def getProb(self, token, sentiment):
        """
            fetch probability using the entire token tuple, not the last one.
        """
        if self.prevBoW is None:
            return self.thisBoW.getTokenCount(token, sentiment) / self.thisBoW.getSentimentCount(sentiment)
        else:
            # return self.thisBoW.getTokenCount(token, sentiment) / self.prevBoW.getTokenCount(token[:-1],sentiment) 
            return self.thisBoW.getTokenCount(token, sentiment) / self.thisBoW.getSentimentCount(sentiment)
