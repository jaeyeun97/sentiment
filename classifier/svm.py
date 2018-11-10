from sklearn.svm import svc

from sklearn.svm import SVC

class SVM(object):
    def __init__(self, trainingSet):
        self.trainingSet = trainingSet
        self.rlookup = dict()
        cnt = 0
        for bow in trainingSet.bows:
            for token in bow.tokenMap.keys():
                self.rlookup[token] = cnt
                cnt += 1

        
