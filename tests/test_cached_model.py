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
import datetime

class TestCachedModel(unittest.TestCase):

    def test_model(self):
        svc = SVC(C=10)
        nsamples = 5000
        nfeatures = 5
        np.random.seed(0)
        X = np.random.sample(size=(nsamples, nfeatures))
        y = np.random.randint(2, size=nsamples)
        start_time = datetime.datetime.now()
        svc.fit(X, y)
        delta_time = datetime.datetime.now() - start_time
        print("fit using %d seconds." % delta_time.seconds) ## roughly 9 seconds
        y_pred = svc.predict(X)

        cached_svc1 = create_cached_model(SVC, C=10)
        start_time = datetime.datetime.now()
        cached_svc1.fit(X, y)
        delta_time = datetime.datetime.now() - start_time
        print("fit using %d seconds." % delta_time.seconds) ## roughly 9 seconds
        cached_y_pred1 = cached_svc1.predict(X)

        cached_svc2 = create_cached_model(SVC, C=10)
        start_time = datetime.datetime.now()
        cached_svc2.fit(X, y)
        delta_time = datetime.datetime.now() - start_time
        print("fit using %d seconds." % delta_time.seconds) ## less than 1 seconds
        cached_y_pred2 = cached_svc2.predict(X)

        self.assertTrue((y_pred == cached_y_pred1).all())
        self.assertTrue((y_pred == cached_y_pred2).all())

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