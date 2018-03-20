# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 23:46:18 2018

@author: jinpeng
"""

import os
import numpy as np
import hashlib

def get_cache_dir():
    cache_dir = os.path.expanduser("~/tmp/talkdata")
    return cache_dir

def get_np_arrays_hash(arrays):
    Xy_hash = ""
    for array in arrays:
        try:
            old_status = array.flags.writeable
            array.flags.writeable = False
            Xy_hash += str(hash(array.data))
            array.flags.writeable = old_status
        except:
            array_del = np.copy(array)
            array_del.flags.writeable = False
            Xy_hash += str(hash(array_del.data))
            del array_del
    Xy_hash = hash(Xy_hash)

def create_cached_model(Model, **kwargs):
    class CachedModel(Model):
        def __init__(self, **kwargs):
            self.cm_hash_key = hash(str(Model) + repr(kwargs))
            self.cm_cache_dir = get_cache_dir()

        def fit(self, X, y):
            Xy_hash = get_np_arrays_hash([X, y])
            fitted_model_hash = str(self.model_hash_key) + str(Xy_hash)
            m = hashlib.md5()
            m.update(fitted_model_hash.encode("utf-8"))
            fitted_model_hash = str(m.hexdigest())

        def predict(self, X):
            pass
    
        def predict_proba(self, X):
            pass

    return CachedModel(kwargs)

