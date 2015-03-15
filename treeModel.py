__author__ = 'alfiya'
import pandas as pd
import re
import preprocess
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import time
import matplotlib.pyplot as plt
import itertools
import statsmodels.api as sm
import matplotlib.pylab as pylab
import cPickle
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from scipy import stats
from sklearn.tree import _tree
from sklearn.utils import array2d
from sklearn.base import BaseEstimator
from functools import partial


def calcCRPS(prob, expected):
    H = (np.tile(np.arange(70), (expected.size, 1)) >= np.tile(expected, (1, 70))).astype(float)
    N = prob.shape[0]
    C = ((prob - H)**2).sum() / (70 * N)
    return C


def corr_matrix(data):
    corr = data.corr()
    with open("data/corr_matrix.pkl", "w") as f:
        cPickle.dump(corr, f)
    plt.figure(figsize=(40, 40))
    fig = sm.graphics.plot_corr(corr.as_matrix(), xnames=list(corr.columns.values), ynames=list(corr.columns.values))
    fig.savefig("data/corr_matrix", format="png")


def getProb(estimator, X):
    pred = estimator.predict(X)
    # pred = np.exp(pred) - 1
    # create probability distribution for each prediction: prob = 0 if x < prediction, prob = 1 otherwise
    # 2 matrices shape (data size, 70)
    # prob = stats.logistic.cdf(x=np.tile(np.arange(70), (pred.size, 1)), loc=(np.tile(pred, (1, 70)) - 2))
    return pred


def scorer(estimator, X, y):
    actual = y
    prob = getProb(estimator, X)
    crps = calcCRPS(prob, actual.reshape((actual.size, 1)))
    # print crps
    return -crps


def intersections(low_null_rate, data):
    for i1 in range(low_null_rate.size):
        # data[low_null_rate[i1] + "_log"] = np.log(data[low_null_rate[i1]]).astype(np.float32)
        # data[low_null_rate[i1] + "_sqrt"] = np.sqrt(data[low_null_rate[i1]]).astype(np.float32)
        # data[low_null_rate[i1] + "_exp"] = np.exp(data[low_null_rate[i1]]).astype(np.float32)
        # data[low_null_rate[i1] + "_q"] = (data[low_null_rate[i1]] * data[low_null_rate[i1]]).astype(np.float32)
        for i2 in range(i1, low_null_rate.size):
            if low_null_rate[i1].split("_")[0] != low_null_rate[i2].split("_")[0] and low_null_rate[i1].split("_")[0] in ["time", "nRadars", "DistanceToRadar"]:
                data[low_null_rate[i1] + "_" + low_null_rate[i2]] = data[low_null_rate[i1]] * data[low_null_rate[i2]]


class classifier(BaseEstimator):

    def __init__(self, outlier_threshold=70., outliers_delete=None, average=np.mean, bias=2., sample_size=1000, tree_params={}, gradientBoostedTree_params = {}):

        self.outlier_threshold = outlier_threshold
        self.outliers_delete = outliers_delete
        self.bias = bias
        self.average = average
        self.sample_size = sample_size
        self.tree_params = tree_params
        self.gradientBoostedTree_params = gradientBoostedTree_params

        # self.estimator = GradientBoostingRegressor(**self.gradientBoostedTree_params)
        self.estimator = None
        self.distribution = None
        self.probabilities = None
        self.n_leaves = None
        self.samples_by_leaf = None
        self.DTYPE = _tree.DTYPE

    def _buildDistribution(self, x, name):
        xk = x.groupby(x).count() / float(x.size)
        return stats.rv_discrete(name="custm {}".format(name), values=(xk.values, xk.index.values))

    def _getProbabilities(self, l, actual):
        if actual[l].size < self.sample_size:
            pred = self.average(actual[l])
            if pred.size > 1:
                pred[pred > self.outlier_threshold] = np.exp(pred[pred > self.outlier_threshold] - self.outlier_threshold) + self.outlier_threshold - 1
            elif pred > self.outlier_threshold:
                pred = np.exp(pred - self.outlier_threshold) + self.outlier_threshold - 1

            out = stats.logistic.cdf(x=np.arange(70), loc=(pred - self.bias))
        else:
            out = self.distribution[l].cdf(np.arange(70))
        return out

    def _cluster(self, X, y):
        y[y > 70] = 70
        self.estimator.fit(X, y)

        # type cast
        if getattr(X, "dtype", None) != self.DTYPE or X.ndim != 2:
            X = array2d(X, dtype=self.DTYPE)

        actual = pd.Series(y, index=self.estimator.tree_.apply(X))
        leaves = np.unique(actual.index.values)

    def fit(self, X, y):
        self.estimator = DecisionTreeRegressor(**self.tree_params)

        if self.outliers_delete is not None:
            outliers = np.where(y > self.outliers_delete)[0]
            y = np.delete(y, outliers)
            X = np.delete(X, outliers, axis=0)
        # print X.shape, y.size
        # y.drop(outliers, inplace=True)
        # X.drop(outliers, inplace=True)
        y[y > self.outlier_threshold] = self.outlier_threshold + np.log(y[y > self.outlier_threshold] - (self.outlier_threshold - 1))
        self.estimator.fit(X, y)

        # type cast
        if getattr(X, "dtype", None) != self.DTYPE or X.ndim != 2:
            X = array2d(X, dtype=self.DTYPE)

        actual = pd.Series(y, index=self.estimator.tree_.apply(X))
        leaves = np.unique(actual.index.values)

        self.n_leaves = leaves.size
        self.samples_by_leaf = pd.Series(actual.index.values).groupby(pd.Series(actual.index.values)).count()

        self.distribution = {l: self._buildDistribution(actual[l], l) for l in leaves}
        self.probabilities = {l: self._getProbabilities(l, actual) for l in leaves}
        return self

    def predict(self, X):
        if getattr(X, "dtype", None) != self.DTYPE or X.ndim != 2:
            X = array2d(X, dtype=self.DTYPE)
        tree_leaves = self.estimator.tree_.apply(X)

        return np.array(map(lambda _: self.probabilities[_], tree_leaves))

    # def set_params(self, **params):
    #     print params
    #     self.bias = params["bias"]
    #     self.average = params["average"]
    #     self.estimator.set_params(**(params["tree_params"]))
    #     return self
    #
    # def get_params(self, **kwargs):
    #     out = {"bias": self.bias, "average": self.average, "tree_params": self.estimator.get_params(**kwargs)}
    #     return out

data = pd.read_csv("data/train_preprocessed.csv")
Id = data["Id"].copy()
data.drop("Id", axis=1, inplace=True)

target = data["Expected"].copy()
print np.percentile(target, 99.55)
# target = np.log(1 + target)
# print target.describe()
# outliers = target.index.values[target.values > 69]
# target.drop(outliers, inplace=True)
# data.drop(outliers, inplace=True)
# print outliers.size, target.size, data.shape

data.drop("Expected", axis=1, inplace=True)
# unique = data.apply(lambda x: x[~pd.isnull(x)].unique().size)
# data.replace([-pd.np.inf], np.finfo(np.float32).min, inplace=True)
# data.replace([pd.np.inf], np.finfo(np.float32).max, inplace=True)

# null stat
# cols = [x for x in list(data.columns.values) if x not in set(preprocess.PRECIP) and not re.findall("_closest", x)]
# null = pd.isnull(data[cols]).sum()/data.shape[0]
#
# data = data.fillna(data.mean())

with open("data/corr_matrix.pkl") as f:
    corr = cPickle.load(f)

# print "******Primary model******"
# start = time.time()
# classifier = ensemble.GradientBoostingRegressor(learning_rate=0.1, n_estimators=30, verbose=1, max_depth=6)
# classifier.fit(data, target)
# importance_primary = pd.Series(classifier.feature_importances_, index=data.columns)
# min_importance = 0.0015
# cols_keep = importance_primary.index.values[importance_primary.values > min_importance]
# print "keep:", cols_keep.size, "total importance: ", importance_primary[importance_primary.values > min_importance].sum(), "from:", importance_primary.sum()
# print "time", round((time.time() - start)/60, 1)
# print cols_keep
# print scorer(classifier, data, target)
corr_cols_drop = corr.columns[~np.in1d(corr.columns.values, data.columns.values)]
corr.drop(corr_cols_drop, axis=0, inplace=True)
corr.drop(corr_cols_drop, axis=1, inplace=True)
threshold = 0.95
indices = np.where((corr > threshold) | (corr < -threshold))
indices = [(corr.index[x], corr.columns[y]) for x, y in zip(*indices) if x != y and x < y]
null = pd.isnull(data).sum()/data.shape[0]
cols_del = np.unique([x if null[x] > null[y] else y for x, y in indices])

print threshold, cols_del.size

data.drop(cols_del, axis=1, inplace=True)
print "dropped correlated features", data.shape

# data = data[cols_keep]
precip_cols = [x for x in list(data.columns.values) if (x in set(preprocess.PRECIP) or re.findall("_closest", x)) and x != "no_echo"]
# data.drop(precip_cols, axis=1, inplace=True)
# print "dropped precip cols", data.shape

# generate intersections
cols = [x for x in list(data.columns.values) if x not in set(preprocess.PRECIP) and not re.findall("_closest", x)]
null = pd.isnull(data[cols]).sum()/data.shape[0]
low_null_rate = null.index[null < 0.1].values
print low_null_rate.size

# low_null_rate = null[cols].index[null[cols] < 0.1].values
# print low_null_rate.size

intersections(low_null_rate, data)

print data.shape
data.replace([pd.np.nan, -pd.np.inf], np.finfo(np.float32).min, inplace=True)
data.replace([pd.np.inf], np.finfo(np.float32).max, inplace=True)
# data = data.fillna(data.mean())

print "******Grid search******"
start = time.time()
estimator = classifier(sample_size=2000000, outlier_threshold=70.9, average=np.mean, bias=2.6, tree_params={"max_depth": 6, "min_samples_leaf": 20})
gridSearch = GridSearchCV(estimator, param_grid={"outlier_threshold": np.arange(69, 80, 0.5)}, scoring=scorer, verbose=True, cv=6, n_jobs=2)
gridSearch.fit(data, target)
scores = gridSearch.grid_scores_

importance = pd.Series(gridSearch.best_estimator_.estimator.feature_importances_, index=data.columns)
importance.sort(ascending=False)
print "time", round((time.time() - start)/60, 1)
print scores
print gridSearch.best_params_, gridSearch.best_score_
print scorer(gridSearch.best_estimator_, data, target)
print importance

best_estimator = gridSearch.best_estimator_
print best_estimator
with open("best_estimator.pkl", "w") as f:
    cPickle.dump(best_estimator, f)
importance.to_csv("importance.csv")

#*********** Submission *************
with open("best_estimator.pkl") as f:
    best_estimator = cPickle.load(f)
test_data = pd.read_csv("data/test_preprocessed.csv")
Id = test_data["Id"].copy()
test_data.drop("Id", axis=1, inplace=True)
# test_data = test_data[cols_keep]
test_data.drop(cols_del, axis=1, inplace=True)
# test_data.drop(precip_cols, axis=1, inplace=True)

intersections(low_null_rate, test_data)

print test_data.shape
test_data.replace([pd.np.nan, -pd.np.inf], np.finfo(np.float32).min, inplace=True)
test_data.replace([pd.np.inf], np.finfo(np.float32).max, inplace=True)
# test_data = test_data.fillna(data.mean())
# test_data.replace(pd.np.nan, -999, inplace=True)

prob = getProb(best_estimator, test_data)

submission = pd.DataFrame(prob, columns=["Predicted"+str(i) for i in range(70)])
submission["Id"] = Id.astype(int)
submission = submission.reindex(columns=(["Id"]+["Predicted"+str(i) for i in range(70)]))
submission.to_csv("submission.csv", index=False)

# [mean: -0.00865, std: 0.00016, params: {'n_estimators': 30, 'subsample': 0.9, 'learning_rate': 0.1, 'max_depth': 6}]
# {'n_estimators': 30, 'subsample': 0.9, 'learning_rate': 0.1, 'max_depth': 6} -0.00864635259266
# -0.00848593007603