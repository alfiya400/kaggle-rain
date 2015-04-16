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
from sklearn.metrics import log_loss
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier


def calcCRPS(prob, expected):
    H = (np.tile(np.arange(70), (expected.size, 1)) >= np.tile(expected, (1, 70))).astype(float)
    N = prob.shape[0]
    C = ((prob - H)**2).sum() / (70 * N)
    return C


def getProb(estimator, X, bias=2.45, outlier_value=25):
    pred = estimator.predict(X)
    cl = estimator.outliers(X)
    zero = (pred < 0.1) & (cl == 0)
    outliers = (pred > 10) & (cl == 2)
    # outliers_val = (pred > outlier_value) & outliers
    # avg = pred[outliers].mean()
    # avg_outliers = (pred > avg) & outliers
    # med = np.median(pred[outliers])
    # med_outliers = (pred > med) & outliers
    # print "25", round(outliers_val.sum()/float(outliers.size), 5), "avg", avg, round(avg_outliers.sum()/float(outliers.size), 5), "median", med, round(med_outliers.sum()/float(outliers.size), 5)
    # outliers = avg_outliers
    pred = pred.reshape((X.shape[0], 1))

    prob = stats.logistic.cdf(x=np.tile(np.arange(70), (pred.size, 1)), loc=(np.tile(pred, (1, 70)) - bias))
    # if zero.any():
    #     prob[zero, :] = np.tile(np.repeat(1, 70), (zero.sum(), 1))
    # if outliers.any():
    #     # prob[outliers, :] = stats.logistic.cdf(x=np.tile(np.arange(70), (outliers.sum(), 1)), loc=(np.tile(outlier_value, (1, 70)) - bias))
    #     prob[outliers, :] = np.tile(np.repeat(0, 70), (outliers.sum(), 1))
    return prob


def scorer(estimator, X, y, bias=2.45, outlier_value=25):
    actual = y
    prob = getProb(estimator, X, bias, outlier_value)
    crps = calcCRPS(prob, actual.reshape((actual.size, 1)))
    print crps
    return -crps


def scorer_class(estimator, X, y):
    pred = estimator.predict(X)
    matr = confusion_matrix(y, pred).astype(float)
    return matr[1, 1]/(matr[0, 1] + matr[1, 0] + matr[1, 1])


def classification_scorer(estimator, X, y):
    pred = estimator.predict_proba(X)
    return -log_loss(y, pred)


class DataTransformer(object):
    def __init__(self, calc_corr=False, corr_file="data/corr_matrix.pkl", corr_threshold=0.98, na_rate_threshold=0.1, na_replace=np.mean,
                 minus_inf_replace=np.finfo(np.float32).min, inf_replace=np.finfo(np.float32).max, scaler=StandardScaler, null_rate=0.9):

        self.corr_file = corr_file
        self.corr_threshold = corr_threshold
        self.na_replace = na_replace
        self.minus_inf_replace = minus_inf_replace
        self.inf_replace = inf_replace
        self.cols_del = None
        self.low_null_rate = None
        self.scaler = scaler()
        self.na_rate_threshold = na_rate_threshold
        self.calc_corr = calc_corr
        self.null_rate = null_rate
        self.high_null_rate = None

    def fit_transform(self, file_in):
        data = pd.read_csv(file_in)
        Id = data["Id"].copy()
        data.drop("Id", axis=1, inplace=True)

        # if self.calc_corr:
        #     corr_matrix(data, corr_file=self.corr_file)

        target = data["Expected"].copy()
        data.drop("Expected", axis=1, inplace=True)

        with open(self.corr_file) as f:
            corr = cPickle.load(f)

        corr_cols_drop = corr.columns[~np.in1d(corr.columns.values, data.columns.values)]
        corr.drop(corr_cols_drop, axis=0, inplace=True)
        corr.drop(corr_cols_drop, axis=1, inplace=True)

        indices = np.where((corr > self.corr_threshold) | (corr < -self.corr_threshold))
        indices = [(corr.index[x], corr.columns[y]) for x, y in zip(*indices) if x != y and x < y]
        null = pd.isnull(data).sum()/data.shape[0]
        self.cols_del = np.unique([x if null[x] > null[y] else y for x, y in indices])

        data.drop(self.cols_del, axis=1, inplace=True)
        print "dropped highly correlated features, new data shape {}".format(data.shape)

        # generate intersections
        cols = [x for x in list(data.columns.values) if x not in set(preprocess.PRECIP) and not re.findall("_closest", x)]
        null = pd.isnull(data[cols]).sum()/data.shape[0]
        self.low_null_rate = null.index[null < self.na_rate_threshold].values
        self.high_null_rate = null.index[null > self.null_rate].values
        data.drop(self.high_null_rate, axis=1, inplace=True)
        print "dropped features with null rate greater {0}, new data shape {1}".format(self.null_rate, data.shape)

        columns = data.columns

        # data = pd.DataFrame(self.scaler.fit_transform(data), columns=columns)
        self._intersections(self.low_null_rate, data)

        print data.shape
        data = data.fillna(np.min(data) - (np.max(data)-np.min(data))/float(data.shape[0]))

        data.replace([-pd.np.inf], self.minus_inf_replace, inplace=True)
        data.replace([pd.np.inf], self.inf_replace, inplace=True)

        data["high_precip"] = data[["graupel", "heavy_rain", "dry_snow", "rain/hail", "moderate_rain"]].values.any(axis=1).astype(np.float32)
        print data["high_precip"].unique()
        return Id, data, target

    def transform(self, file_in):
        data = pd.read_csv(file_in)
        Id = data["Id"].copy()
        data.drop("Id", axis=1, inplace=True)
        data.drop(self.cols_del, axis=1, inplace=True)
        data.drop(self.high_null_rate, axis=1, inplace=True)
        columns = data.columns
        # data = data.fillna(self.na_replace(data))
        # data = pd.DataFrame(self.scaler.transform(data), columns=columns)
        self._intersections(self.low_null_rate, data)
        data = data.fillna(np.min(data) - (np.max(data)-np.min(data))/float(data.shape[0]))

        data.replace([-pd.np.inf], self.minus_inf_replace, inplace=True)
        data.replace([pd.np.inf], self.inf_replace, inplace=True)

        data["high_precip"] = data[["graupel", "heavy_rain", "dry_snow", "rain/hail", "moderate_rain"]].values.any(axis=1).astype(np.float32)
        print "generated from {0} data frame shape {1}".format(file_in, data.shape)

        return Id, data

    @staticmethod
    def _intersections(low_null_rate, data, intersect_cols={"time", "nRadars", "DistanceToRadar"}):
        for i1 in range(low_null_rate.size):
            # data[low_null_rate[i1] + "_log"] = np.log(1 + data[low_null_rate[i1]]).astype(np.float32)
            # data[low_null_rate[i1] + "_sqrt"] = np.sqrt(data[low_null_rate[i1]]).astype(np.float32)
            # data[low_null_rate[i1] + "_exp"] = np.exp(data[low_null_rate[i1]]).astype(np.float32)
            # data[low_null_rate[i1] + "_q"] = (data[low_null_rate[i1]] * data[low_null_rate[i1]] * data[low_null_rate[i1]]).astype(np.float32)
            for i2 in range(i1, low_null_rate.size):
                spl1 = low_null_rate[i1].split("_")  # if low_null_rate[i1] not in preprocess.PRECIP else [low_null_rate[i1]]
                spl2 = low_null_rate[i2].split("_")  # if low_null_rate[i2] not in preprocess.PRECIP else [low_null_rate[i2]]

                if spl1[0] != spl2[0] and spl1[0] in intersect_cols and (spl1[1] == spl2[1] if len(spl1)+len(spl2) == 4 else True): #
                    data[low_null_rate[i1] + "_" + low_null_rate[i2]] = data[low_null_rate[i1]] * data[low_null_rate[i2]]


class regressor(GradientBoostingRegressor):
    def fit(self, X, y, monitor=None):
        # na_rows = np.isnan(X).any(axis=1)
        # X = X[~na_rows]
        # y = y[~na_rows]

        threshold = 69.0
        outliers = y > threshold
        not_zero = (y > 0) & (y <= threshold)
        y_class = y.copy()
        y_class[outliers] = 2
        y_class[not_zero] = 1
        outliersNum = outliers.sum()
        nonZeroNum = not_zero.sum()
        zeroNum = y_class.size - outliersNum - nonZeroNum
        cl = RandomForestClassifier(max_depth=6, min_samples_leaf=20, n_estimators=10)
        sample_weight = np.ones(y_class.shape, dtype=int)
        weight1 = 1  # max(round(0.2 * zeroNum / nonZeroNum), 1)
        weight2 = max(round(0.05 * zeroNum / outliersNum), 1)
        sample_weight[not_zero] = weight1
        sample_weight[outliers] = weight2
        # train, test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=10)
        # y_outliers = y > 69
        cl.fit(X, y_class)
        print classification_scorer(cl, X, y_class)
        # test = np.append(test, cl.predict(test).reshape((test.shape[0], 1)), 1)
        # X = np.append(X, cl.predict(X).reshape((X.shape[0], 1)), 1)
        y[y > 70.] = 75.
        # y = np.log(1 + y)

        self.classifier = cl

        return super(regressor, self).fit(X, y, monitor)

    def predict(self, X):
        # X = X.copy()
        # m = np.nanmedian(X, axis=0)
        # # print m.shape
        # for i in np.arange(X.shape[1]):
        #     na = np.isnan(X[:, i])
        #     X[na, i] = m[i]
        pred = super(regressor, self).predict(X)
        # pred = np.exp(pred) - 1
        return pred  # super(regressor, self).predict(X)

    def outliers(self, X):
        # X = X.copy()
        # m = np.nanmedian(X, axis=0)
        # for i in np.arange(X.shape[1]):
        #     na = np.isnan(X[:, i])
        #     X[na, i] = m[i]
        return self.classifier.predict(X)


class AdaBoostRegr(AdaBoostRegressor):
    def fit(self, X, y, monitor=None):
        y[y > 70.9] = 70.9
        # y[y > 71] = 71 + np.log(y[y > 71] - (71 - 1))
        return super(AdaBoostRegr, self).fit(X, y, monitor)


class init_cl:
    def __init__(self, est):
        self.est = est

    def predict(self, X):
        return self.est.predict(X)[:, np.newaxis]

    def fit(self, X, y):
        self.est.fit(X, y)


if __name__ == "__main__":
    data_transformer = DataTransformer()
    Id, data, target = data_transformer.fit_transform("data/train_preprocessed.csv")

    print "******Grid search******"
    start = time.time()
    print target.median(), (target - target.median()).median(), (target - target.median()).mean()

    n_estimators = 40
    estimator = regressor(max_depth=7, min_samples_leaf=20, learning_rate=0.05, n_estimators=n_estimators, subsample=0.8, random_state=0)
    # estimator = regressor(normalize=True) #RANSACRegressor(random_state=0, min_samples=0.1, residual_threshold=70)
    gridSearch = GridSearchCV(estimator, param_grid={}, scoring=partial(scorer, bias=2.45), verbose=True, cv=6, n_jobs=2)
    gridSearch.fit(data.values, target.values)
    scores = gridSearch.grid_scores_
    # estimator.fit(data.values, target.values)
    # cumsum = -np.cumsum(estimator.oob_improvement_)
    # x = np.arange(n_estimators) + 1
    # p = plt.plot(x, cumsum, label='OOB loss')
    # plt.savefig("oob_score.pdf", format="pdf")
    columns = list(data.columns)  # + ["outlier"]
    importance = pd.Series(gridSearch.best_estimator_.feature_importances_, index=columns)
    importance.sort(ascending=False)
    print "time", round((time.time() - start)/60, 1)
    print scores
    print gridSearch.best_params_, gridSearch.best_score_
    print scorer(gridSearch.best_estimator_, data.values, target.values)
    print importance
    print confusion_matrix(target > 69, gridSearch.best_estimator_.predict(data.values) > 69)
    cPickle.dump(gridSearch.best_estimator_, open("best_estimator.pkl", "w"))
    importance.to_csv("importance.csv")

    # for _i in np.arange(5):
    #     print _i, "\n", confusion_matrix(target > _i, np.round(best_estimator.predict(data)) > _i)

    #*********** Submission *************

    best_estimator = cPickle.load(open("best_estimator.pkl"))
    # best_outlier_value = 0
    # best_score = -1
    # for outlier_value in np.arange(2, 3, 0.1):
    #     score = scorer(best_estimator, data, target, bias=outlier_value)
    #     if score > best_score:
    #         best_score = score
    #         best_outlier_value = outlier_value
    #     # print outlier_value, scorer(gridSearch.best_estimator_, data, target, outlier_value=outlier_value)
    # print best_outlier_value, best_score

    Id, test_data = data_transformer.transform("data/test_preprocessed.csv")

    prob = getProb(best_estimator, test_data.values) #, outlier_value=best_outlier_value)

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

#{} -0.00859428712097
#0.00845132124168

#
# {} -0.00856565905373
# 1186
# 0.00841950355553