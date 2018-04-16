cached_sklearn
==============

It is the same idea with [joblib memory](https://pythonhosted.org/joblib/memory.html), but cached_sklearn is specified for the sklearn estimators.
. When we tune the parameters of estimators, 
you will probably run serverals times of estimators with same parameters. 
However, each `fit` will takes a lot of time. This package can save computing
time when several `fit`s and `predict`s with same parameters are called.

Here is an example:

```python
from cached_sklearn.model import create_cached_model
from sklearn.svm import SVC
from sklearn.base import clone
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import datetime

Model = SVC
kwargs = {'C': 10, 'probability': True, 'random_state': 0}

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
y_pred_proba = clf.predict_proba(X)
y_pred = clf.predict(X)

cached_clf1 = create_cached_model(Model, **kwargs)
start_time = datetime.datetime.now()
cached_clf1.fit(X, y)
delta_time = datetime.datetime.now() - start_time
print("fit using %d seconds." % delta_time.seconds) ## roughly more than 1 seconds
cached_y_pred1 = cached_clf1.predict(X)
cached_y_pred_proba1 = cached_clf1.predict_proba(X)

cached_clf2 = create_cached_model(Model, **kwargs)
start_time = datetime.datetime.now()
cached_clf2.fit(X, y)
delta_time = datetime.datetime.now() - start_time
print("fit using %d seconds." % delta_time.seconds) ## less than 1 seconds
cached_y_pred2 = cached_clf2.predict(X)

print((y_pred == cached_y_pred1).all())
print((y_pred == cached_y_pred2).all())
print((y_pred_proba == cached_y_pred_proba1).all())
```

By default, all the cached files are stored in the `~/.cached_sklearn`. You can remove all the cached if you need.
