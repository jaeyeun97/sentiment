import numpy as np
from scipy.stats import binom
from scipy.special import comb

class SignTest(object):
    def __init__(self, actual):
        self.actual = actual

    def test(self, baseline, target):
        assert len(baseline) == len(target)
        null = 0
        plus = 0
        minus = 0
        for i in range(len(baseline)):
            if baseline[i] == target[i]:
                null += 1
            elif baseline[i] == self.actual[i]:
                minus += 1
            elif target[i] == self.actual[i]:
                plus += 1

        N = 2 * np.ceil(null/2) + plus + minus
        k = np.ceil(null/2) + np.minimum(plus, minus)
        q = 0.5
        return 2 * binom.cdf(k, N, q)
