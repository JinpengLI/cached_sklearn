# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 23:46:18 2018

@author: jinpeng
"""

import os
import numpy as np
import hashlib
import joblib
import copy

def get_cache_dir():
    cache_dir = os.path.expanduser("~/.cached_sklearn/cached")
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
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
    return Xy_hash

def create_cached_model(Model, **kwargs):
    class CachedModel(Model):
        def __init__(self, **kwargs):
            self._cm_hash_key = hash(str(Model) + repr(kwargs))
            self._cm_cache_dir = get_cache_dir()
            self._cm_Model = Model
            super(CachedModel, self).__init__(**kwargs)

        def fit(self, X, y, **kwargs):
            array_hash = get_np_arrays_hash([X, y])
            fitted_model_hash = str(self._cm_hash_key) + str(array_hash) + str(kwargs)
            m = hashlib.md5()
            m.update(fitted_model_hash.encode("utf-8"))
            fitted_model_hash = str(m.hexdigest())
            path_trained_model_dir = os.path.join(self._cm_cache_dir, fitted_model_hash)
            path_trained_model = os.path.join(path_trained_model_dir, "model.joblib")
            if os.path.isfile(path_trained_model):
                cached_model = joblib.load(path_trained_model)
                for attr_name in cached_model.__dict__:
                    try:
                        setattr(self, attr_name, getattr(cached_model, attr_name))
                    except:
                        pass
            else:
                super(CachedModel, self).fit(X, y, **kwargs)
                if not os.path.isdir(path_trained_model_dir):
                    os.makedirs(path_trained_model_dir)
                raw_model = Model()
                for attr_name in self.__dict__:
                    try:
                        setattr(raw_model, attr_name, copy.deepcopy(getattr(self, attr_name)))
                    except:
                        pass
                joblib.dump(raw_model, path_trained_model)


    return CachedModel(**kwargs)

