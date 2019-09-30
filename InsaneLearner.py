

import BagLearner as bl
import LinRegLearner as lrl

class InsaneLearner(object):

    def __init__(self, verbose=False):
        self.learner = bl.BagLearner(learner=bl.BagLearner, kwargs={"learner": lrl.LinRegLearner, "kwargs": {}, "bags": 20, "boost": False, "verbose": verbose}, bags=20, boost=False, verbose=verbose)

    def author(self):
        return "nwatt3"

    def addEvidence(self, dataX, dataY):
        self.learner.addEvidence(dataX, dataY)

    def query(self, points):
        return self.learner.query(points)