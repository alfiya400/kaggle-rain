__author__ = 'alfiya'
import pandas as pd
import re
import preprocess
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier
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
from functools import partial
from sklearn.metrics import confusion_matrix


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


def getProb(estimator, X, bias=2.45, outlier_value=6):
    pred = estimator.predict(X)
    outliers = (pred > outlier_value) & (estimator.outliers(X).astype(bool))
    print outliers.sum()
    pred = pred.reshape((X.shape[0], 1))
    # create probability distribution for each prediction: prob = 0 if x < prediction, prob = 1 otherwise
    # 2 matrices shape (data size, 70)
    prob = stats.logistic.cdf(x=np.tile(np.arange(70), (pred.size, 1)), loc=(np.tile(pred, (1, 70)) - bias))
    # if outliers.any():
    #     # print outliers.sum(), prob[outliers, :].shape, np.tile(np.repeat(0, 70), (outliers.sum(), 1)).shape
    #     # prob[outliers, :] = stats.logistic.cdf(x=np.tile(np.arange(70), (outliers.sum(), 1)), loc=(np.tile(outlier_value, (1, 70)) - bias))
    #     prob[outliers, :] = np.tile(np.repeat(0, 70), (outliers.sum(), 1))
    return prob


def scorer(estimator, X, y, bias=2.45, outlier_value=6):
    actual = y
    prob = getProb(estimator, X, bias, outlier_value)
    crps = calcCRPS(prob, actual.reshape((actual.size, 1)))
    print crps
    return -crps


def scorer_class(estimator, X, y):
    pred = estimator.predict(X)
    matr = confusion_matrix(y, pred).astype(float)
    return matr[1, 1]/(matr[0, 1] + matr[1, 0] + matr[1, 1])


def intersections(low_null_rate, data, intersect_cols={"time", "nRadars", "DistanceToRadar"}):
    for i1 in range(low_null_rate.size):
        # data[low_null_rate[i1] + "_log"] = np.log(1 + data[low_null_rate[i1]]).astype(np.float32)
        # data[low_null_rate[i1] + "_sqrt"] = np.sqrt(data[low_null_rate[i1]]).astype(np.float32)
        # data[low_null_rate[i1] + "_exp"] = np.exp(data[low_null_rate[i1]]).astype(np.float32)
        # data[low_null_rate[i1] + "_q"] = (data[low_null_rate[i1]] * data[low_null_rate[i1]]).astype(np.float32)
        for i2 in range(i1, low_null_rate.size):
            spl1 = low_null_rate[i1].split("_")
            spl2 = low_null_rate[i2].split("_")
            if spl1[0] != spl2[0] and spl1[0] in intersect_cols and (spl1[1] == spl2[1] if len(spl1)+len(spl2) == 4 else True):
                data[low_null_rate[i1] + "_" + low_null_rate[i2]] = data[low_null_rate[i1]] * data[low_null_rate[i2]]


class regressor(GradientBoostingRegressor):
    def fit(self, X, y, monitor=None):
        # outliers = np.where(y > 70.)[0]
        # y = np.delete(y, outliers)
        # X = np.delete(X, outliers, axis=0)
        cl = GradientBoostingClassifier(learning_rate=0.1, n_estimators=25, max_depth=6, subsample=0.9)
        # train, test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=10)

        y_outliers = y > 69
        cl.fit(X, y_outliers)
        # print scorer_class(cl, train, y_outliers), scorer_class(cl, test, y_test > 69)
        # test = np.append(test, cl.predict(test).reshape((test.shape[0], 1)), 1)
        # X = np.append(X, cl.predict(X).reshape((X.shape[0], 1)), 1)
        y[y > 70.] = 70.
        # y[y > 71] = 71 + np.log(y[y > 71] - (71 - 1))
        self.classifier = cl
        return super(regressor, self).fit(X, y, monitor)

    def predict(self, X):
        # X = np.append(X, self.classifier.predict(X).reshape((X.shape[0], 1)), 1)
        return super(regressor, self).predict(X)

    def outliers(self, X):
        return self.classifier.predict(X)


class AdaBoostRegr(AdaBoostRegressor):
    def fit(self, X, y, monitor=None):
        y[y > 70.9] = 70.9
        # y[y > 71] = 71 + np.log(y[y > 71] - (71 - 1))
        return super(AdaBoostRegr, self).fit(X, y, monitor)

data = pd.read_csv("data/train_preprocessed.csv")
Id = data["Id"].copy()
data.drop("Id", axis=1, inplace=True)

target = data["Expected"].copy()
print np.percentile(target, 99.65)
data.drop("Expected", axis=1, inplace=True)

# target = np.log(1 + target)
# print target.describe()
# outliers = target.index.values[target.values > 70]
# target.drop(outliers, inplace=True)
# data.drop(outliers, inplace=True)
# print outliers.size, target.size, data.shape

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
print low_null_rate
# low_null_rate = null[cols].index[null[cols] < 0.1].values
# print low_null_rate.size

intersections(low_null_rate, data)

print data.shape
data.replace([pd.np.nan, -pd.np.inf], np.finfo(np.float32).min, inplace=True)
data.replace([pd.np.inf], np.finfo(np.float32).max, inplace=True)
# data = data.fillna(data.mean())

print "******Grid search******"
start = time.time()
# cl = GradientBoostingClassifier(learning_rate=0.1, n_estimators=10, max_depth=6, subsample=0.9)
# target_outliers = target > 69
#
# gridSearch = GridSearchCV(cl, param_grid={"n_estimators": [15]}, scoring=scorer_class, verbose=True, cv=6, n_jobs=2)
# gridSearch.fit(data, target_outliers)
# print "time", round((time.time() - start)/60, 1)
# print gridSearch.grid_scores_
# print gridSearch.best_params_, gridSearch.best_score_
# print scorer_class(gridSearch.best_estimator_, data, target_outliers)
# print confusion_matrix(target_outliers, gridSearch.best_estimator_.predict(data))


# estimator = AdaBoostRegr(base_estimator=DecisionTreeRegressor(max_depth=6, min_samples_leaf=20), learning_rate=0.1, n_estimators=30, loss="exponential")
estimator = regressor(max_depth=6, min_samples_leaf=20, learning_rate=0.1, n_estimators=30, subsample=0.9)
gridSearch = GridSearchCV(estimator, param_grid={}, scoring=partial(scorer, bias=2.45), verbose=True, cv=6, n_jobs=2)
gridSearch.fit(data.values, target.values)
scores = gridSearch.grid_scores_

columns = list(data.columns) # + ["outlier"]
importance = pd.Series(gridSearch.best_estimator_.feature_importances_, index=columns)
importance.sort(ascending=False)
print "time", round((time.time() - start)/60, 1)
print scores
print gridSearch.best_params_, gridSearch.best_score_
print scorer(gridSearch.best_estimator_, data, target)
print importance
print confusion_matrix(target > 69, gridSearch.best_estimator_.predict(data) > 69)
cPickle.dump(gridSearch.best_estimator_, open("best_estimator.pkl", "w"))
importance.to_csv("importance.csv")

# estimator = classifier_ada(base_estimator=DecisionTreeRegressor(max_depth=6, min_samples_leaf=20), loss="exponential", learning_rate=0.1, n_estimators=30)
# estimator.fit(data, target)
# print confusion_matrix(target > 69, estimator.predict(data) > 69)
# for outlier_value in np.arange(70, 80, 0.5):
#     print outlier_value, scorer(estimator, data, target, outlier_value=outlier_value)

# 1.7 -0.00927094966718
# start = time.time()
# estimator = classifier(subsample=0.9, max_depth=12, learning_rate=0.1, n_estimators=30, min_samples_leaf=20, alpha=0.5, loss="quantile")
# estimator.fit(data, target)
# print round((time.time()-start)/60)
#
# estimator = classifier(subsample=0.9, max_depth=6, learning_rate=0.1, n_estimators=30, min_samples_leaf=20)
# estimator.fit(data, target)
# for bias in np.arange(2.4, 2.6, 0.01):
#     print bias, scorer(estimator, data, target, bias)

#*********** Submission *************

best_estimator = cPickle.load(open("best_estimator.pkl"))
best_outlier_value = 0
best_score = -1
for outlier_value in np.arange(10, 70, 5):
    score = scorer(best_estimator, data, target, outlier_value=outlier_value)
    if score > best_score:
        best_score = score
        best_outlier_value = outlier_value
    # print outlier_value, scorer(gridSearch.best_estimator_, data, target, outlier_value=outlier_value)
print best_outlier_value, best_score

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

prob = getProb(best_estimator, test_data, outlier_value=best_outlier_value)

submission = pd.DataFrame(prob, columns=["Predicted"+str(i) for i in range(70)])
submission["Id"] = Id.astype(int)
submission = submission.reindex(columns=(["Id"]+["Predicted"+str(i) for i in range(70)]))
submission.to_csv("submission.csv", index=False)

# [mean: -0.00865, std: 0.00016, params: {'n_estimators': 30, 'subsample': 0.9, 'learning_rate': 0.1, 'max_depth': 6}]
# {'n_estimators': 30, 'subsample': 0.9, 'learning_rate': 0.1, 'max_depth': 6} -0.00864635259266
# -0.00848593007603

# [mean: -0.00856, std: 0.00014, params: {'min_samples_leaf': 5}, mean: -0.00858, std: 0.00015, params: {'min_samples_leaf': 10}, mean: -0.00857, std: 0.00014, params: {'min_samples_leaf': 15}, mean: -0.00857, std: 0.00014, params: {'min_samples_leaf': 20}, mean: -0.00859, std: 0.00014, params: {'min_samples_leaf': 30}, mean: -0.00859, std: 0.00014, params: {'min_samples_leaf': 40}, mean: -0.00861, std: 0.00013, params: {'min_samples_leaf': 50}, mean: -0.00863, std: 0.00014, params: {'min_samples_leaf': 70}, mean: -0.00868, std: 0.00015, params: {'min_samples_leaf': 100}]
# {'min_samples_leaf': 5} -0.00855780192158
# -0.00840568993988

# {'min_samples_leaf': 20} -0.00859139571852
# -0.00847649868772

# [mean: -0.00848, std: 0.00014, params: {'min_samples_leaf': 20}]
# {'min_samples_leaf': 20} -0.00847926589781
# -0.00826568739242

# [mean: -0.00857, std: 0.00014, params: {'min_samples_leaf': 20}]
# {'min_samples_leaf': 20} -0.0085716686151
# -0.00841192988038