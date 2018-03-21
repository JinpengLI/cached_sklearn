# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 17:40:45 2018

@author: jinpeng
"""

import unittest
from cached_sklearn.model import create_cached_model
from sklearn.svm import SVC
from sklearn.base import clone
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import datetime

class TestCachedModel(unittest.TestCase):

    def test_model(self):
        Model = GradientBoostingClassifier
        kwargs = {'learning_rate': 0.1}
        self._sub_test(Model, kwargs)

    def _sub_test(self, Model, kwargs):
        clf = Model(**kwargs)

        nsamples = 5000
        nfeatures = 5
        np.random.seed(0)
        X = np.random.sample(size=(nsamples, nfeatures))
        y = np.random.randint(2, size=nsamples)
        start_time = datetime.datetime.now()
        clf.fit(X, y)
        delta_time = datetime.datetime.now() - start_time
        print("fit using %d seconds." % delta_time.seconds) ## roughly more than 1 seconds
        y_pred = clf.predict(X)
        y_pred_proba = clf.predict_proba(X)

        cached_svc1 = create_cached_model(Model, **kwargs)
        if len(kwargs.keys()) > 0:
            cached_svc1_params = cached_svc1.get_params()
            param_key = kwargs.keys()[0]
            self.assertTrue(cached_svc1_params[param_key] == kwargs[param_key])

        start_time = datetime.datetime.now()
        cached_svc1.fit(X, y)
        delta_time = datetime.datetime.now() - start_time
        print("fit using %d seconds." % delta_time.seconds) ## roughly more than 1 seconds
        cached_y_pred1 = cached_svc1.predict(X)
        cached_y_pred_proba1 = cached_svc1.predict_proba(X)

        cached_svc2 = create_cached_model(Model, **kwargs)
        start_time = datetime.datetime.now()
        cached_svc2.fit(X, y)
        delta_time = datetime.datetime.now() - start_time
        print("fit using %d seconds." % delta_time.seconds) ## less than 1 seconds
        cached_y_pred2 = cached_svc2.predict(X)

        self.assertTrue((y_pred == cached_y_pred1).all())
        self.assertTrue((y_pred == cached_y_pred2).all())
        self.assertTrue((y_pred_proba == cached_y_pred_proba1).all())

#    def test_copy(self):
#        svc1 = SVC(C=10)
#        nsamples = 100
#        nfeatures = 5
#        X = np.random.sample(size=(nsamples, nfeatures))
#        y = np.random.randint(2, size=nsamples)
#        svc1.fit(X, y)
#        y_pred1 = svc1.predict(X)
#
#        svc2 = SVC()
#        svc2 = clone(svc1)
#        y_pred2 = svc2.predict(X)
#        
#        self.assertTrue((y_pred1 == y_pred2).all())
        
        
if __name__ == '__main__':
    unittest.main()