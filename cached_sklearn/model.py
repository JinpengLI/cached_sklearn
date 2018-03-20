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
        if array is None:
            continue
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
            self._cm_cache_dir = get_cache_dir()
            self._cm_Model = Model
            self._cm_fit_hash = None
            super(CachedModel, self).__init__(**kwargs)

        def get_params(self, ):
            model = Model()
            params = model.get_params()
            new_params = {}
            for key in params:
                new_params[key] = getattr(self, key)
            return new_params

        def get_hash_model_params(self, ):
            return hash(str(self._cm_Model) + repr(self.get_params()))

        def get_hash_fit_model_params(self, X, y, kwargs):
            array_hash = get_np_arrays_hash([X, y])
            fitted_model_hash = str(self.get_hash_model_params()) + str(array_hash) + str(kwargs)
            m = hashlib.md5()
            m.update(fitted_model_hash.encode("utf-8"))
            fitted_model_hash = str(m.hexdigest())
            return fitted_model_hash

        def get_hash_predict(self, X, **kwargs):
            if self._cm_fit_hash is not None:
                hash_predict = str(self._cm_fit_hash) + str(get_np_arrays_hash([X])) + str(kwargs)
                m = hashlib.md5()
                m.update(hash_predict.encode("utf-8"))
                hash_predict = str(m.hexdigest())
                return hash_predict
            return None

        def fit(self, X, y=None, **kwargs):
            self._cm_fit_hash = self.get_hash_fit_model_params(X, y, kwargs)            
            path_trained_model_dir = os.path.join(self._cm_cache_dir, self._cm_fit_hash)
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

        def predict(self, X, **kwargs):
            hash_predict = self.get_hash_predict(X, **kwargs)
            if hash_predict is None:
                return super(CachedModel, self).predict(X, **kwargs)

            path_predict_dir = os.path.join(self._cm_cache_dir, hash_predict)
            if not os.path.isdir(path_predict_dir):
                os.makedirs(path_predict_dir)
            path_predict_file = os.path.join(path_predict_dir, "predict.npy")
            if os.path.isfile(path_predict_file):
                return np.load(path_predict_file)
            else:
                predict_res = super(CachedModel, self).predict(X, **kwargs)
                np.save(path_predict_file, predict_res)
                return predict_res

    return CachedModel(**kwargs)

