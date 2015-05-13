__author__ = 'alfiya'
import pandas as pd
from pandas.tools.plotting import radviz, andrews_curves
import re
import preprocess
import numpy as np
import time
import cPickle
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from scipy import stats
from functools import partial
from sklearn.metrics import confusion_matrix, log_loss, adjusted_rand_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import KernelPCA, PCA
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from pandas.tools.plotting import table
from sklearn.pipeline import Pipeline

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
randomState = np.random.RandomState(15)

def timePassed(start):
    return round((time.time() - start) / 60)


def calcCRPS(prob, expected):
    H = (np.tile(np.arange(70), (expected.size, 1)) >= np.tile(expected, (1, 70))).astype(float)
    N = prob.shape[0]
    C = ((prob - H)**2).sum() / (70 * N)
    return C


def getProb(pred, bias=2.45, scale=1):
    pred = pred.reshape((pred.size, 1))
    prob = stats.logistic.cdf(x=np.tile(np.arange(70), (pred.size, 1)), loc=(np.tile(pred, (1, 70)) - bias), scale=scale)
    return prob


def getPred(pred, weights):
    def getPredbyRow(x, weights=None):
        return np.concatenate([k * w for k, w in zip(x, weights)])

    return np.apply_along_axis(getPredbyRow, axis=1, arr=pred, weights=weights)


def scorer(estimator, X, y, bias=2.45, scale=1,
           classif=False, weights=None,
           has_labels=False, bias_scale_by_label=None):
    actual = y.reshape((y.size, 1))

    # if classif is True then estimator returns matrix of probabilities
    # otherwise - regression task and estimator return a predicted value
    if classif:  # classification
        pred = estimator.predict_proba(X)
        # if has_labels is True then use different weights for each cluster
        if has_labels:
            labels = estimator.labels(X)
            pred_ = np.ones((X.shape[0], 71), dtype=np.float32)

            for l in np.unique(labels):
                pred_[labels == l] = getPred(pred[l], weights=weights[l]) #["bias"], scale=bias_scale_by_label[l]["scale"])
        else:
            pred_ = getPred(pred, weights=weights)
        # print pred_.shape
        prob = np.cumsum(pred_[:, :-1], axis=1)
        # print prob
    else:  # regression
        pred = estimator.predict(X)
        # if has_labels is True then use best bias and scale for each cluster
        if has_labels:
            labels = estimator.labels(X)
            prob = np.ones((X.shape[0], 70), dtype=np.float32)

            for l in np.unique(labels):
                prob[labels == l] = getProb(pred[labels == l], **bias_scale_by_label[l]) #["bias"], scale=bias_scale_by_label[l]["scale"])
        else:
            prob = getProb(pred, bias=bias, scale=scale)

    crps = calcCRPS(prob, actual)

    return -crps


def scorer_class(estimator, X, y):
    pred = estimator.predict(X)
    matr = confusion_matrix(y, pred).astype(float)
    return matr[1, 1]/(matr[0, 1] + matr[1, 0] + matr[1, 1])


def classification_scorer(estimator, X, y):
    pred = estimator.predict_proba(X)
    return -log_loss(y, pred)


class DataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, corr_file="data/corr_matrix.pkl", corr_threshold=0.999,
                 low_na_rate_threshold=0.1, high_na_rate_threshold=0.9,
                 minus_inf_replace=np.finfo(np.float32).min, inf_replace=np.finfo(np.float32).max):

        self.columns = columns

        self.corr_file = corr_file
        self.corr_threshold = corr_threshold

        self.low_na_rate_threshold = low_na_rate_threshold
        self.high_na_rate_threshold = high_na_rate_threshold

        self.minus_inf_replace = minus_inf_replace
        self.inf_replace = inf_replace
        self.new_columns = columns

        self.cols_del = None
        self.low_null_rate = None
        self.high_null_rate = None
        self.na_replace = None

    def fit(self, X, y=None):

        data = pd.DataFrame(X, columns=self.columns)
        with open(self.corr_file) as f:
            corr = cPickle.load(f)

        corr_cols_drop = corr.columns[~np.in1d(corr.columns.values, data.columns.values)]
        corr.drop(corr_cols_drop, axis=0, inplace=True)
        corr.drop(corr_cols_drop, axis=1, inplace=True)

        indices = np.where((corr > self.corr_threshold) | (corr < -self.corr_threshold))
        indices = [(corr.index[x], corr.columns[y]) for x, y in zip(*indices) if x != y and x < y]
        null = pd.isnull(data).sum() / data.shape[0]
        self.cols_del = np.unique([x if null[x] > null[y] else y for x, y in indices])

        data.drop(self.cols_del, axis=1, inplace=True)
        # print "dropped highly correlated features, new data shape {}".format(data.shape)

        # generate intersections
        cols = [x for x in list(data.columns.values) if x not in set(preprocess.PRECIP) and not re.findall("_closest", x)]
        null = pd.isnull(data[cols]).sum()/data.shape[0]
        self.low_null_rate = null.index[null < self.low_na_rate_threshold].values
        self.high_null_rate = null.index[null > self.high_na_rate_threshold].values
        data.drop(self.high_null_rate, axis=1, inplace=True)
        # print "dropped features with null rate greater {0}, new data shape {1}".format(self.high_na_rate_threshold, data.shape)

        intersect_columns = self._intersections(self.low_null_rate, data)

        # print "added intersections data shape {0}".format(data.shape)

        self.na_replace = np.min(data) - (np.max(data) - np.min(data)) / float(data.shape[0])
        self.new_columns = self.new_columns[~np.in1d(self.new_columns, np.concatenate((self.cols_del, self.high_null_rate)))]
        self.new_columns = np.concatenate((self.new_columns, intersect_columns))
        return self

    def transform(self, X):
        data = pd.DataFrame(X, columns=self.columns)
        data.drop(self.cols_del, axis=1, inplace=True)
        data.drop(self.high_null_rate, axis=1, inplace=True)

        _ = self._intersections(self.low_null_rate, data)

        data = data.fillna(self.na_replace)

        data.replace([-pd.np.inf], self.minus_inf_replace, inplace=True)
        data.replace([pd.np.inf], self.inf_replace, inplace=True)

        # print "final data shape {0}".format(data.shape)

        return data.values, self.new_columns

    @staticmethod
    def _intersections(low_null_rate, data, intersect_cols={"time", "DistanceToRadar"}):
        intersect_columns = []
        for i1 in range(low_null_rate.size):
            # data[low_null_rate[i1] + "_log"] = np.log(1 + data[low_null_rate[i1]]).astype(np.float32)
            # data[low_null_rate[i1] + "_sqrt"] = np.sqrt(data[low_null_rate[i1]]).astype(np.float32)
            # data[low_null_rate[i1] + "_exp"] = np.exp(data[low_null_rate[i1]]).astype(np.float32)
            # data[low_null_rate[i1] + "_q"] = (data[low_null_rate[i1]] * data[low_null_rate[i1]] * data[low_null_rate[i1]]).astype(np.float32)
            for i2 in range(i1, low_null_rate.size):
                spl1 = low_null_rate[i1].split("_")
                spl2 = low_null_rate[i2].split("_")

                if spl1[0] != spl2[0] and spl1[0] in intersect_cols and (spl1[1] == spl2[1] if len(spl1)+len(spl2) == 4 else True):
                    data[low_null_rate[i1] + "_" + low_null_rate[i2]] = data[low_null_rate[i1]] * data[low_null_rate[i2]]
                    intersect_columns.append(low_null_rate[i1] + "_" + low_null_rate[i2])
        return intersect_columns


class Clusterer(BaseEstimator):
    def __init__(self, clusterer=None, n_clusters=2, columns=None, cluster_columns=None, clusterer_=None, na_replace=None):
        self.clusterer = clusterer
        self.n_clusters = n_clusters
        self.columns = columns
        self.cluster_columns = cluster_columns
        self.clusterer_ = clusterer_
        self.na_replace = na_replace

    def _cluster_columns(self, columns):
        if columns is None:
            self.cluster_columns = None
        else:
            features = set()
            clust_cols = []
            clust_cols_ix = []
            for i, _ in enumerate(columns):
                if _.split("_")[0] not in features and (len(_.split("_")) <= 2) and re.findall("mean|min", _):
                    features.add(_.split("_")[0])
                    clust_cols.append(_)
                    clust_cols_ix.append(i)
            self.cluster_columns = clust_cols_ix
            # print clust_cols

    def fit_transform(self, X):
        self.na_replace = np.nanmin(X, axis=0) - (np.nanmax(X, axis=0) - np.nanmin(X, axis=0)) / float(X.shape[0])
        if np.isnan(X).any():
            X = np.where(np.isnan(X), self.na_replace, X)

        if self.clusterer is None:
            self.clusterer_ = KMeans(n_clusters=self.n_clusters)
        else:
            self.clusterer_ = globals()[self.clusterer](n_clusters=self.n_clusters)

        self._cluster_columns(self.columns)

        if self.cluster_columns is None:
            labels = self.clusterer_.fit_predict(X)
        else:
            labels = self.clusterer_.fit_predict(X.copy()[:, self.cluster_columns])
        return labels

    def transform(self, X):
        if np.isnan(X).any():
            X = np.where(np.isnan(X), self.na_replace, X)

        if self.cluster_columns is None:
            labels = self.clusterer_.predict(X)
        else:
            labels = self.clusterer_.predict(X.copy()[:, self.cluster_columns])
        return labels

class Regressor(BaseEstimator):
    def __init__(self, est=None, params=None, class_as_cluster=False, cl=None, classifier_params=None):
        self.est = est
        self.params = params
        self.class_as_cluster = class_as_cluster
        self.cl = cl
        self.classifier_params = classifier_params

    def _init_params(self, data_size, data_width, clusters):
        # if data_size < 40000:
        #     depth = 3
        # else:
        #     depth = 6
        if self.est is None:
            self.est = GradientBoostingRegressor
        if self.params is None:
            self.params = {"max_depth": 6, "min_samples_leaf": 20, "learning_rate": 0.05, "n_estimators": 40, "subsample": 0.8, "random_state":0}
        if self.clust is None:
            self.clust = KMeans

        if self.cl is None:
            self.cl = RandomForestClassifier
        if self.classifier_params is None:
            self.classifier_params = {"max_depth": 6, "min_samples_leaf": 20, "random_state": 0}

        self.classifier = self.cl(**self.classifier_params)
        self.regressors = {c: self.est(**self.params) for c in clusters}
        self.columns = []
        self.n_clusters = len(clusters)

    def _fit_models(self, X, y, monitor, labels):
        for c in np.unique(labels):  # iter through clusters
            ix = np.where(labels == c)[0]
            self.regressors[c].fit(X[ix, :], y[ix], monitor=monitor)
            # self.classifiers[c].fit(X[ix, :], y[ix] > 1)

    def _predict(self, X, labels):
        pred = np.zeros(shape=(X.shape[0],), dtype=float)  # init
        for c in np.unique(labels):  # iter through clusters
            ix = np.where(labels == c)[0]
            pred[ix] = self.regressors[c].predict(X[ix, :])
        return pred

    def fit(self, X, y, monitor=None):
        self._init_params(X["data"].shape[0], X["data"].shape[1], np.unique(X["labels"]))
        y[y > 70.] = 70.

        self._fit_models(X["data"], y, monitor, X["labels"])

        #     y_class = np.zeros(shape=y.shape, dtype=np.int8)
        #     y_class[((y >= 1))] = 1
        #     # y_class[y > 69] = 2
        #     print np.bincount(y_class)
        #     X_train, X_test, y_train, y_test, y_class_train, y_class_test =\
        #         train_test_split(X, y, y_class, test_size=0.7, random_state=0)
        #
        #     sample_weight = np.ones(shape=y_train.shape)
        #     sample_weight[((y_train >= 1))] = 5
        #     # sample_weight[y_train > 69] = 10
        #
        #     self.classifier.fit(X_train, y_class_train, sample_weight)
        #     labels = self.classifier.predict(X_test)
        #     print np.bincount(labels)
        #     self._fit_models(X_test, y_test, monitor, labels)
        return self

    def predict(self, X):
        return self._predict(X["data"], X["labels"])

    @property
    def feature_importances_(self):
        return sum(regr.feature_importances_ for regr in self.regressors.itervalues()) / self.n_clusters


class myPredictor(BaseEstimator):
    def __init__(self, cluster=None, columns=None, dataTransformer=None,
                 model_params=None, base_predictor="GBRT", bins=None, subsample=100000):
        self.cluster = cluster
        self.columns = columns
        self.dataTransformer = dataTransformer
        self.model_params = model_params
        self.base_predictor = base_predictor
        self.bins = bins
        self.subsample = subsample

    def fit(self, X, y):
        self.model = {}
        if self.bins is None:
            self.bins = {0: None, 1: None}
        labels = self.cluster["cluster"].transform(X)

        for l in np.unique(labels):
            lX, columns_ = self.dataTransformer[l].transform(X[labels == l])
            ly = y[labels == l]

            self.model[l] = Pipeline([("featureSelector", FeatureSelector()),
                                      ("regressor", globals()[self.base_predictor]())
                                      ])
            self.model[l].set_params(**self.model_params[l])
            self.model[l].set_params(**dict(regressor__verbose=1))
            self.model[l].fit(lX, ly, regressor__subsample=self.subsample, regressor__bins=self.bins[l])

        return self

    def predict(self, X):
        labels = self.cluster["cluster"].transform(X)
        pred = np.zeros(shape=(X.shape[0],), dtype=float)
        for l in np.unique(labels):
            lX, columns_ = self.dataTransformer[l].transform(X[labels == l])
            pred[labels == l] = self.model[l].predict(lX)
        return pred

    def predict_proba(self, X):
        labels = self.cluster["cluster"].transform(X)
        pred = {}
        for l in np.unique(labels):
            lX, columns_ = self.dataTransformer[l].transform(X[labels == l])
            pred[l] = self.model[l].predict_proba(lX)
        return pred

    def labels(self, X):
        return self.cluster["cluster"].transform(X)


class GBRT(GradientBoostingRegressor):
    def fit(self, X, y, monitor=None, subsample=50000, bins=None, **kwargs):
        if subsample and (X.shape[0] > subsample):
            size = min(X.shape[0], subsample)
            ix = randomState.choice(np.arange(X.shape[0]), size=size, replace=False)
            tmpX = X[ix, :]
            tmpY = y[ix]
            return super(GBRT, self).fit(tmpX, tmpY, monitor=monitor)
        else:
            return super(GBRT, self).fit(X, y, monitor=monitor)


class GBCT(GradientBoostingClassifier):
    def fit(self, X, y, monitor=None, subsample=50000, bins=None, **kwargs):
        # print X.shape, self.get_params()
        y = np.ceil(y).astype(np.int8)
        y = np.vectorize(lambda x: bins[x])(y)
        if subsample and (X.shape[0] > subsample):
            # size = min(X.shape[0], subsample)
            ix = randomState.choice(np.arange(X.shape[0]), size=subsample, replace=False)
            tmpX = X[ix, :]
            tmpY = y[ix]
            return super(GBCT, self).fit(tmpX, tmpY, monitor=monitor)
        else:
            return super(GBCT, self).fit(X, y, monitor=monitor)


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importance=None, importance_threshold=None, column_names=None):
        self.feature_importance = feature_importance
        self.importance_threshold = importance_threshold
        self.column_names = column_names
        self.important_features = None
        self.important_features_names = None

    def fit(self, X, y=None, subsample=False, bins=None):
        self.important_features = np.where(self.feature_importance > self.importance_threshold)[0]
        self.important_features_names = self.column_names[self.important_features]
        # print "important features {0}".format(self.important_features.size), self.feature_importance[self.important_features].sum(), self.feature_importance.sum()

        return self

    def transform(self, X):
        return X[:, self.important_features]

    @property
    def important_columns_(self):
        return self.important_features_names


def findBestBiasAndScale(X, y, estimator, randomState):
    best_scale, best_bias, best_score = 0., 0., -1.
    ix = randomState.choice(np.arange(X.shape[0]), size=100000, replace=False)
    tmpX, tmpY = X[ix, :], y[ix].copy()
    tmpY[tmpY > 70.] = 70.

    if X.shape[0] < 500000:
        scale_range = np.arange(1.5, 1.7, 0.1)
        bias_range = np.arange(2.1, 2.7, 0.1)
    else:
        scale_range = np.arange(3, 5, 0.5)
        bias_range = np.arange(12., 17., 1.)

    for scale in scale_range:
        for bias in bias_range:
            score = scorer(estimator, tmpX, tmpY, scale=scale, bias=bias)
            print scale, bias, score
            if score > best_score:
                best_score, best_scale, best_bias = score, scale, bias
    print best_scale, best_bias, best_score
    return best_scale, best_bias, best_score

if __name__ == "__main__":
    CLASSIF = False

    if CLASSIF:
        BASE_PREDICTOR = "GradientBoostingClassifier"
        my_predictor = "GBCT"
    else:
        BASE_PREDICTOR = "GradientBoostingRegressor"
        my_predictor = "GBRT"

    # Load data
    data = pd.read_csv("data/train_preprocessed.csv")

    Id = data["Id"].copy()
    data.drop("Id", axis=1, inplace=True)
    target = data["Expected"].copy().values
    data.drop("Expected", axis=1, inplace=True)

    columns = data.columns.values
    data = data.values
    # Clustering
    clust = Clusterer(clusterer="KMeans", n_clusters=2, columns=columns)
    labels = clust.fit_transform(data)

    print "******Grid search******"
    output = {"clusterer": clust, "columns": columns, "clust": {}}
    # randomState = np.random.RandomState(15)
    pred = np.zeros(shape=target.shape, dtype=float)
    for l in np.unique(labels):
        n = (labels == l).sum()
        dataTransformer = DataTransformer(columns=columns, corr_threshold=0.98)
        X, columns_ = dataTransformer.fit_transform(data[labels == l])
        y = target[labels == l]
        y[y > 70.] = 70.
        big_cluster = X.shape[0] > 500000
        print X.shape
        start = time.time()
        e = globals()[BASE_PREDICTOR](max_depth=6 if big_cluster else 3, learning_rate=0.05, n_estimators=10, min_samples_leaf=20, random_state=0)
        print e
        scorer_ = partial(scorer, bias=13. if big_cluster else 2.3, scale=4. if big_cluster else 1.6)
        param_grid = dict(featureSelector__importance_threshold=[0.001, 0.01, 0.02, 0.03],
                          regressor__max_depth=[6 if big_cluster else 3])
        e.fit(X, y)
        print "primary score {0}, time {1}".format(scorer_(e, X, y), timePassed(start))
        prediction_model = Pipeline([("featureSelector", FeatureSelector(e.feature_importances_, 0.004, columns_)),
                                    ("regressor", globals()[my_predictor](min_samples_leaf=20, learning_rate=0.05, n_estimators=100, subsample=0.8, random_state=0))
                                     ])
        gridSearch = GridSearchCV(prediction_model, param_grid=param_grid, scoring=scorer_, verbose=True, cv=6, n_jobs=2)
        gridSearch.fit(X, y)

        # output
        importance = pd.Series(gridSearch.best_estimator_.steps[-1][-1].feature_importances_, index=gridSearch.best_estimator_.steps[0][-1].important_columns_)
        importance.sort(ascending=False)
        print "GS scores: {0}".format(gridSearch.grid_scores_)
        print "GS best params: {0}, score {1}".format(gridSearch.best_params_, gridSearch.best_score_)
        print "features {0}, train score {1}".format(importance.size, scorer_(gridSearch.best_estimator_, X, y))
        print importance
        print confusion_matrix(y > 1, gridSearch.best_estimator_.predict(X) > 1)

        pred[labels == l] = gridSearch.best_estimator_.predict(X)

        best_scale, best_bias, best_score = findBestBiasAndScale(X, y, gridSearch.best_estimator_, randomState)

        output["clust"][l] = {"dataTransformer": dataTransformer,
                              "model": gridSearch.best_estimator_,
                              "importance": importance,
                              "scores": gridSearch.grid_scores_,
                              "best_params": gridSearch.best_params_,
                              "best_distr_par": {"scale": best_scale, "bias": best_bias}
                              }

    prob = np.ones((data.shape[0], 70), dtype=np.float32)
    best_par = {k: v["best_distr_par"] for k, v in output["clust"].iteritems()}
    for l in np.unique(labels):
        prob[labels == l] = getProb(pred[labels == l], **best_par[l])
    print "total score: ", calcCRPS(prob, target.reshape((target.size, 1)))
    cPickle.dump(output, open("best_model.pkl", "w"))

    del X, y, prob, pred, importance, e, prediction_model, output, dataTransformer, columns_, clust, labels

    # *********** Submission *************
    best_output = cPickle.load(open("best_model.pkl"))

    # MODEL VALIDATION
    model = myPredictor(cluster={"cluster": best_output["clusterer"]}, columns=best_output["columns"],
                        dataTransformer={k: v["dataTransformer"] for k, v in best_output["clust"].iteritems()},
                        model_params={k: v["model"].get_params() for k, v in best_output["clust"].iteritems()},
                        base_predictor=my_predictor)

    scorer_ = partial(scorer, has_labels=True, best_par={k: v["best_distr_par"] for k, v in best_output["clust"].iteritems()})
    gridSearch = GridSearchCV(model, param_grid={}, scoring=scorer_, verbose=True, cv=2, n_jobs=1)
    gridSearch.fit(data, target)
    print "GS scores: {0}".format(gridSearch.grid_scores_)
    print "train score {0}".format(scorer_(gridSearch.best_estimator_, data, target))

    cPickle.dump(gridSearch.best_estimator_, open("best_regressor.pkl", "w"))

    test_data = pd.read_csv("data/test_preprocessed.csv")

    Id = test_data["Id"].copy()
    test_data.drop("Id", axis=1, inplace=True)

    test_data = test_data.values
    pred = gridSearch.best_estimator_.predict(test_data)

    best_par = {k: v["best_distr_par"] for k, v in best_output["clust"].iteritems()}
    labels = gridSearch.best_estimator_.labels(test_data)

    prob = np.ones((test_data.shape[0], 70), dtype=np.float32)

    for l in np.unique(labels):
        prob[labels == l] = getProb(pred[labels == l], bias=best_par[l]["bias"], scale=best_par[l]["scale"])

    submission = pd.DataFrame(prob, columns=["Predicted"+str(i) for i in range(70)])
    submission["Id"] = Id.astype(int)
    submission = submission.reindex(columns=(["Id"]+["Predicted"+str(i) for i in range(70)]))
    submission.to_csv("submission.csv", index=False)

# GS scores: [mean: -0.00825, std: 0.00006, params: {}]
# train score -0.00811924743582
