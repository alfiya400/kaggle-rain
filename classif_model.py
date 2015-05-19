__author__ = 'alfiya'
import re
import time
import cPickle
import datetime

import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

import preprocess


randomState = np.random.RandomState(15)
N_CLASSES = 71  # there will be 71 classes (0, 1, 2... 68, 69, 70)
N_CUMUL_CLASSES = 70  # for CRPS score only 70 classes are considered (0, 1, ..., 69)
MAX_VALUE = 70.  # class 70 will represent all values greater 69, so I will stack all values greater 70 in 70
IMPORTANCE_ESTIMATOR = "FeatureImportanceEstimator"  # estimator for feature importances
BASE_ESTIMATOR = "GBCT"  # estimator for classification


def time_passed(start):
    return round((time.time() - start) / 60)


def load_data(filename, test=False):
    data = pd.read_csv(filename)

    Id = data["Id"].copy()
    data.drop("Id", axis=1, inplace=True)

    if not test:
        target = data["Expected"].copy().values
        # stack all values greater MAX_VALUE into MAX_VALUE
        target[target > MAX_VALUE] = MAX_VALUE
        data.drop("Expected", axis=1, inplace=True)

    return (data.values, Id, data.columns.values) if test else (data.values, Id, target, data.columns.values)


def get_bins(target, min_bin_count):
    """
        Combines target values into bins with minimal count as @min_bin_count
    :param target: numpy.array, dtype = int
    :param min_bin_count: int
        Minimum frequency for each bin
    :return: dict
        Dictionary with binId for each class and with class weights for each bin, weights sum up to 1 for each bin
        Structure:
            dict(bins=<list of binID, list size=N_CLASSES>,
                weights=<list of weights of classes for each bin, list size=number of bins>)
    """
    bc = np.bincount(target)  # class frequencies
    bins_, bin_count, bin_number = [], 0., 0
    bin_weights = [np.array([])]  # weight of each class in every bin (sums to 1 for each bin)
    for v in bc:  # iter through class frequencies
        # fill bin
        bin_count += v  # bin freq
        bins_.append(bin_number)  # add new bin ID
        bin_weights[bin_number] = np.append(bin_weights[bin_number], v)

        if bin_count > min_bin_count:  # if exceed min threshold => close current bin, open new bin
            bin_weights[bin_number] /= bin_count

            bin_count = 0
            bin_number += 1
            bin_weights.append(np.array([]))

    # close last bin
    if len(bin_weights[-1]):
        bin_weights[-1] /= bin_count
    else:
        bin_weights.pop()

    # print bins frequencies
    print np.bincount(np.vectorize(lambda x: bins_[x])(target))

    return dict(bins=bins_, weights=bin_weights)


def calc_crps(prob, expected):
    """
    Calculates Continuous Ranked Probability Score
    The smaller the better
    https://www.kaggle.com/c/how-much-did-it-rain/details/evaluation

    :param prob: numpy array, shape (n samples, 70)
        Cumulative probability
    :param expected: numpy array, shape (n samples,)
        Actual value
    :return: float
        Continuous Ranked Probability Score
    """
    H = (np.tile(np.arange(N_CUMUL_CLASSES), (expected.size, 1)) >= np.tile(expected, (1, N_CUMUL_CLASSES))).astype(float)
    N = prob.shape[0]
    C = ((prob - H) ** 2).sum() / (N_CUMUL_CLASSES * N)
    return C


def scorer(estimator, X, y):
    """
    Scorer function
    Used to evaluate model quality
    :param estimator: object
        estimator used for predictions, must have predict_proba method
    :param X: numpy.array
    :param y: numpy.array
    :return: float
    """
    actual = y.reshape((y.size, 1))
    pred = estimator.predict_proba(X)  # predicted probabilities for each class
    prob = np.cumsum(pred[:, :-1], axis=1)  # cumulative probabilities
    crps = calc_crps(prob, actual)
    print crps
    return -crps


class DataTransformer(BaseEstimator, TransformerMixin):
    """
    DataTransformer
        - keeps only one feature from 2 highly correlated features (kepps the one with lower nan rate)
        - removes features with high nan rate
        - adds intersections between "time", "DistanceToRadar" and features with low nan rate
        - fill nan with large negative constant value
    """
    def __init__(self, columns, corr_file="data/corr_matrix.pkl", corr_threshold=0.98,
                 low_na_rate_threshold=0.1, high_na_rate_threshold=0.9,
                 neg_inf_replace=np.finfo(np.float32).min, pos_inf_replace=np.finfo(np.float32).max):
        """

        :param columns: numpy array
            Columns names
        :param corr_file:
            cPickle file name with correlations pandas.DataFrame
        :param corr_threshold:
            threshold for high correlations
        :param low_na_rate_threshold: float
        :param high_na_rate_threshold: float
        :param neg_inf_replace: float
            value to replace nan and -inf
        :param pos_inf_replace: float
            value to replace inf
        :return:
        """

        self.columns = columns

        self.corr_file = corr_file
        self.corr_threshold = corr_threshold

        self.low_na_rate_threshold = low_na_rate_threshold
        self.high_na_rate_threshold = high_na_rate_threshold

        self.neg_inf_replace = neg_inf_replace
        self.pos_inf_replace = pos_inf_replace
        self.na_replace = neg_inf_replace  # value to replace nan (same as -inf)

        self.new_columns = columns  # column names for new dataset after transformations
        self.cols_del = None  # columns to delete (the one that have high correlation)
        self.cols_del_ix = None
        self.low_null_rate = None  # columns with low nan rate
        self.low_null_rate_ix = None
        self.high_null_rate = None  # columns with high nan rate
        self.high_null_rate_ix = None

    def fit(self, X, y=None):
        """
        Fit DataTransformer
        :param X: numpy.array
        :param y:
        :return:
        """

        # load correlations to pandas.DataFrame
        with open(self.corr_file) as f:
            corr = cPickle.load(f)

        # consider only columns from data file (in case if columns in correlations differs from columns in data)
        corr_cols_drop = corr.columns[~np.in1d(corr.columns.values, self.columns)]
        corr.drop(corr_cols_drop, axis=0, inplace=True)
        corr.drop(corr_cols_drop, axis=1, inplace=True)

        # find highly correlated features
        indices = np.where((corr > self.corr_threshold) | (corr < -self.corr_threshold))

        # find columns to delete
        indices = [(corr.index[x], corr.columns[y]) for x, y in zip(*indices) if x != y and x < y]
        null = pd.Series(np.isnan(X).sum(axis=0) / float(X.shape[0]), index=self.columns)
        self.cols_del = np.unique([x if null[x] > null[y] else y for x, y in indices])
        self.cols_del_ix = np.where(np.in1d(self.columns, self.cols_del))[0]

        null.drop(self.cols_del, inplace=True)

        # find columns with high and low nan rates
        cols = [x for x in list(null.index.values) if x not in set(preprocess.PRECIP)]  # ignore hydrometeor type columns
        null = null[cols]

        self.low_null_rate = null[null < self.low_na_rate_threshold].index.values
        self.low_null_rate_ix = np.where(np.in1d(self.columns, self.low_null_rate))[0]

        self.high_null_rate = null[null > self.high_na_rate_threshold].index.values
        self.high_null_rate_ix = np.where(np.in1d(self.columns, self.high_null_rate))[0]

        # generate columns names for intersections
        intersect_columns = self.__intersections(X, get_columns=True)

        # new column names
        self.new_columns = np.concatenate((np.delete(self.new_columns,
                                                     np.append(self.cols_del_ix, self.high_null_rate_ix)),
                                           intersect_columns))
        # print self.low_null_rate, self.low_null_rate_ix
        # print self.high_null_rate, self.high_null_rate_ix
        return self

    def transform(self, X):
        """
        Transform data
        :param X: numpy.array
        :return:
            Returns transformed data and column names for new data
        """
        X = self.__intersections(X, get_columns=False)  # generate intersections
        # drop columns with high correlation and high nan rate
        X = np.delete(X, np.append(self.cols_del_ix, self.high_null_rate_ix), axis=1)
        # nan, -inf, inf replace
        X = np.where(np.isnan(X), self.na_replace, X)
        X = np.where(np.isposinf(X), self.pos_inf_replace, X)
        X = np.where(np.isneginf(X), self.neg_inf_replace, X)

        return X, self.new_columns

    def __intersections(self, X, get_columns=True, intersect_cols={"time", "DistanceToRadar"}):
        """
        Generate intersections between @intersect_cols and self.low_null_rate columns
        Omit intersections between same type of features
            (e.g. no intersections like (DistanceToRadar_min & DistanceToRadar_min), (RR1_mean & RR1_std) etc..)
        Omit intersections between different type of aggregations
            (e.g. no intersections like (DistanceToRadar_min & RR1_mean),
            for DistanceToRadar_min and RR1's aggregations only (DistanceToRadar_min & RR1_min) will be generated )

        :param X: numpy.array
            Data
        :param get_columns: bool
            Whether to get generated column names or generate intersections itself
        :param intersect_cols: set
        :return: list
            List of names of generated columns
        """
        intersect_columns = []
        for i1, c1 in enumerate(self.low_null_rate):
            for i2, c2 in zip(range(i1 + 1, self.low_null_rate.size), self.low_null_rate[i1 + 1:]):
                spl1 = c1.split("_")
                spl2 = c2.split("_")

                if spl1[0] != spl2[0] and spl1[0] in intersect_cols and (spl1[1] == spl2[1] if len(spl1)+len(spl2) == 4 else True):
                    # data[low_null_rate[i1] + "_" + low_null_rate[i2]] = data[low_null_rate[i1]] * data[low_null_rate[i2]]
                    if get_columns:
                        intersect_columns.append(c1 + "_" + c2)
                    else:
                        X = np.append(X, (X[:, self.low_null_rate_ix[i1]] * X[:, self.low_null_rate_ix[i2]]).reshape((-1, 1)), axis=1)

        return intersect_columns if get_columns else X


class Clusterer(BaseEstimator):
    """
    Model used for clusterization

    """
    def __init__(self, clusterer="KMeans", n_clusters=2, columns=None, cluster_columns=None, na_replace=None):
        """

        :param clusterer: str (optional)
            Clusterer name (should be in globals())
        :param n_clusters: int (optional)
            Number of clusters
        :param columns: numpy.array (optional)
            Column names for data
        :param cluster_columns: numpy.array (optional)
            Columns indices used for clusterization.
            If None and @columns is None then all columns are used,
            If None and @columns is not None then columns with "mean" ("min" for DistanceToRadar) aggregation are used
            If not None then @columns should be defined too
        :param na_replace:
            Values to replace nan
        :return:
        """
        self.clusterer = clusterer
        self.n_clusters = n_clusters
        self.columns = columns
        self.cluster_columns = cluster_columns
        self.na_replace = na_replace

    def __cluster_columns(self, columns):
        """
        Finds columns indices used for cluster analysis
        If @columns is None then all columns are used,
        If @columns is not None then columns with "mean" ("min" for DistanceToRadar) aggregation are used,
            intersections columns and hydrometeor types are ignored
        :param columns: numpy.array
            data column names
        :return:
        """

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

    def fit_transform(self, X):
        """
        Replace nan with value (min - small step), small step = (max - min)/n_samples
        Fit and predict cluster labels

        :param X: numpy.array
        :return:
        """
        # nan replace
        self.na_replace = np.nanmin(X, axis=0) - (np.nanmax(X, axis=0) - np.nanmin(X, axis=0)) / float(X.shape[0])
        if np.isnan(X).any():
            X = np.where(np.isnan(X), self.na_replace, X)

        # define clusterer
        self.clusterer_ = globals()[self.clusterer](n_clusters=self.n_clusters)

        # find columns to be used in cluster analysis
        if self.cluster_columns is None:
            self.__cluster_columns(self.columns)

        # fit clusterer
        if self.cluster_columns is None:
            labels = self.clusterer_.fit_predict(X)
        else:
            labels = self.clusterer_.fit_predict(X.copy()[:, self.cluster_columns])
        return labels

    def transform(self, X):
        # replace nan
        if np.isnan(X).any():
            X = np.where(np.isnan(X), self.na_replace, X)

        if self.cluster_columns is None:
            labels = self.clusterer_.predict(X)
        else:
            labels = self.clusterer_.predict(X.copy()[:, self.cluster_columns])
        return labels


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Feature Selector transformer
        Select only features with importance greater than importance threshold

    """
    def __init__(self, feature_importance=None, importance_threshold=None, column_names=None):
        """

        :param feature_importance: numpy.array
            Feature importances, must sum up to 1.
        :param importance_threshold: float
            Minimum feature importance
        :param column_names:
            Data columns names
        """
        self.feature_importance = feature_importance
        self.importance_threshold = importance_threshold
        self.column_names = column_names
        self.important_features = None
        self.important_features_names = None

    def fit(self, X=None, y=None):
        self.important_features = np.where(self.feature_importance > self.importance_threshold)[0]
        self.important_features_names = self.column_names[self.important_features]

        return self

    def transform(self, X):
        return X[:, self.important_features]

    @property
    def important_columns_(self):
        return self.important_features_names


def get_class_prob_from_bin_prob(pred, weights):
    """
        Calculates probabilities for classes from probabilities of bins
        Probability of each bin distributes across classes proportional to class frequencies
            (so the bigger class frequency the bigger the probability of this class)

    :param pred: numpy.array, shape (n samples, n bins)
    :param weights: list of numpy.array, list size = number of bins
        Weights of classes for each bin
    :return: numpy.array, shape (n_samples, n classes)
    """

    def get_prob_by_row(x, weights=None):
        return np.concatenate([bin_prob * class_weights for bin_prob, class_weights in zip(x, weights)])

    return np.apply_along_axis(get_prob_by_row, axis=1, arr=pred, weights=weights)


class GBCT(XGBClassifier):
    """
        Wrapper around XGBClassifier

        Overwritten methods:

        Fit:
        -----
            Allows build model on subsample of data
            Includes transformation of raw target variable to bins of classes

        Predict_proba:
        ---------------
            Returns matrix of probabilities per class, instead of per bins

    """
    def fit(self, X, y, subsample=50000, bins=None):
        """

        :param X: numpy.array
            Data
        :param y: numpy.array
            Target
        :param subsample: int
            Number of samples to fit model
        :param bins: map
            Map {"bins": <list of binId>, list size = N_CLASSES,
                "weights": <list of numpy.array>, list size = number of bins, each numpy.array contains classes weights in bin
            }
        :return: :raise Exception:
        """

        y = np.ceil(y).astype(np.int8)
        if bins is None:
            raise Exception("bins should be not None")
        else:
            self.bins = bins

        y = np.vectorize(lambda x: self.bins["bins"][x])(y)
        if subsample and (X.shape[0] > subsample):
            # size = min(X.shape[0], subsample)
            ix = randomState.choice(np.arange(X.shape[0]), size=subsample, replace=False)
            tmpX = X[ix, :]
            tmpY = y[ix]
            return super(GBCT, self).fit(tmpX, tmpY)
        else:
            return super(GBCT, self).fit(X, y)

    def predict_proba(self, X):
        """

        :param X: numpy.array
        :return: numpy.array, shape (n samples, n classes)
            Returns matrix of probabilities per class, instead of per bins
        """
        pred = super(GBCT, self).predict_proba(X)
        return get_class_prob_from_bin_prob(pred, weights=self.bins["weights"])


class FeatureImportanceEstimator(DecisionTreeClassifier):
    """FeatureImportanceEstimator
        Wrapper around DecisionTreeClassifier

        Overwritten methods:

        Fit:
        -----
            Allows build model on subsample of data
            Includes transformation of raw target variable to bins of classes

        Predict_proba:
        ---------------
            Returns matrix of probabilities per class, instead of per bins

    """
    def fit(self, X, y, sample_weight=None, subsample=50000, bins=None):
        """

        :param X: numpy.array
            Data
        :param y: numpy.array
            Target
        :param sample_weight: numpy.array, shape (n samples,)
            Sample weights
        :param subsample: int
            Number of samples to fit model
        :param bins: map
            Map {"bins": <list of binId>, list size = N_CLASSES,
                "weights": <list of numpy.array>, list size = number of bins, each numpy.array contains classes weights in bin
            }
        :return: :raise Exception:
        """
        y = np.ceil(y).astype(np.int8)
        if bins is None:
            raise Exception("bins should be not None")
        else:
            self.bins = bins

        y = np.vectorize(lambda x: self.bins["bins"][x])(y)
        if subsample and (X.shape[0] > subsample):
            # size = min(X.shape[0], subsample)
            ix = randomState.choice(np.arange(X.shape[0]), size=subsample, replace=False)
            tmpX = X[ix, :]
            tmpY = y[ix]
            return super(FeatureImportanceEstimator, self).fit(tmpX, tmpY, sample_weight=sample_weight)
        else:
            return super(FeatureImportanceEstimator, self).fit(X, y, sample_weight=sample_weight)

    def predict_proba(self, X, **kwargs):
        """

        :param X: numpy.array
        :param kwargs:
        :return: numpy.array, shape (n samples, n classes)
            Returns matrix of probabilities per class, instead of per bins
        """
        pred = super(FeatureImportanceEstimator, self).predict_proba(X)
        return get_class_prob_from_bin_prob(pred, weights=self.bins["weights"])


class MyPredictor(BaseEstimator):
    """
    Estimator for all data
        Split data to clusters
        Transforms data chunks
        Builds model
    """
    def __init__(self, cluster, columns, dataTransformer,
                 model, bins, base_predictor="GBCT", subsample=100000):
        """

        :param cluster: dict
            {"cluster": Clusterer object}
        :param columns: numpy.array
            Data column names
        :param dataTransformer: dict
            Dictionary of Data Transformers for each cluster
            Structure:
                {cluster_label: DataTransformer object for cluster_label in list_of_clusters}
        :param model: dict
            Dictionary of Estimators for each cluster
            Structure:
                {cluster_label: prediction model for cluster_label in list_of_clusters}
        :param bins: dict
            Dictionary of bin maps for each cluster
            Structure:
                {cluster_label: {"bins": <list of binId>, "weights": <list of class weights>}
                    for cluster_label in list_of_clusters}
        :param base_predictor:

        :param subsample: int
            Subsample size to fit model
        """
        self.cluster = cluster
        self.columns = columns
        self.dataTransformer = dataTransformer
        self.model = model
        # self.base_predictor = base_predictor
        self.bins = bins
        self.subsample = subsample

    def fit(self, X, y):
        """

        :param X: numpy.array
        :param y: numpy.array
        :return:
        """

        labels = self.cluster["cluster"].transform(X)

        for l in np.unique(labels):
            lX, columns_ = self.dataTransformer[l].transform(X[labels == l])
            ly = y[labels == l]

            # self.model[l] = Pipeline([("featureSelector", FeatureSelector()),
            #                           ("regressor", globals()[self.base_predictor]())
            #                           ])
            # self.model[l].set_params(**self.model_params[l])
            self.model[l].fit(lX, ly, estimator__subsample=self.subsample, estimator__bins=self.bins[l])
            print "trained", datetime.datetime.now()

        return self

    def predict_proba(self, X):
        """

        :param X: numpy.array
        :return: numpy.array, shape (n_samples, N_CLASSES)
        """
        labels = self.cluster["cluster"].transform(X)  # clusters

        pred = np.ones((X.shape[0], N_CLASSES), dtype=float)
        for l in np.unique(labels):
            lX, columns_ = self.dataTransformer[l].transform(X[labels == l])  # transform data
            pred[labels == l] = self.model[l].predict_proba(lX)  # predict

        return pred

    def labels(self, X):
        """
            Predicts cluster labels for each sample
        :param X: numpy.array
        :return:
        """
        return self.cluster["cluster"].transform(X)


def grid_search(datafile, gs_output, subsample_size=50000, cv=6, n_jobs=2):
    """
        Performs hyperparameters grid search and dumps best results to @outputfile
    :param datafile: str
        Path to a data file
    :param gs_output: str
        cPickle-Filename for grid search output
    :param subsample_size: int
        subsample size to fit model
    :param cv: int
        Number of folds in cross-validation
    :param n_jobs: int
        Number of jobs to run in parallel
    """
    # Load data
    data, Id, target, columns = load_data(datafile, test=False)

    # Clustering
    clust = Clusterer(clusterer="KMeans", n_clusters=2, columns=columns)
    labels = clust.fit_transform(data)

    # Join classes into bigger ones
    bins = {}
    for l in np.unique(labels):
        bins[l] = get_bins(np.ceil(target[labels == l]).astype(np.int8),
                           min_bin_count=4000 if (labels == l).sum() > 500000 else 3000)

    # Hyperparameters grid search for each cluster
    print "******Grid search******"
    output = {"clusterer": clust, "columns": columns, "clust": {}, "bins": bins}
    pred = np.ones((data.shape[0], N_CLASSES), dtype=np.float32)
    # iterate through clusters
    for l in np.unique(labels):
        # transform data
        data_transformer_params = dict(corr_file="data/corr_matrix.pkl", corr_threshold=0.98,
                                       low_na_rate_threshold=0.1, high_na_rate_threshold=0.9,
                                       neg_inf_replace=np.finfo(np.float32).min,
                                       pos_inf_replace=np.finfo(np.float32).max)
        dataTransformer = DataTransformer(columns=columns, **data_transformer_params)
        X, columns_ = dataTransformer.fit_transform(data[labels == l])
        y = target[labels == l]
        print X.shape
        # there are only 2 clusters, I use this indicator to set different parameters for each cluster
        big_cluster = X.shape[0] > 500000

        # Feature Importances
        start = time.time()
        importance_estimator_params = dict(min_samples_leaf=20, random_state=0)
        e = globals()[IMPORTANCE_ESTIMATOR](**importance_estimator_params)
        e.fit(X, y, bins=bins[l], subsample=False)
        print "primary score {0}, time {1}".format(scorer(e, X, y), time_passed(start))

        # Prediction model
        predictor_params = dict(min_child_weight=20)
        prediction_model = Pipeline([("featureSelector", FeatureSelector(e.feature_importances_, 0.001, columns_)),
                                    ("estimator", globals()[BASE_ESTIMATOR](**predictor_params))
                                     ])

        # set a param grid to find best hyperparameters
        param_grid = dict(featureSelector__importance_threshold=[0.001],
                          estimator__max_depth=[5] if big_cluster else [3],
                          estimator__subsample=[0.8], estimator__n_estimators=[100],
                          estimator__learning_rate=[0.1])
        gridSearch = GridSearchCV(prediction_model, param_grid=param_grid, scoring=scorer,
                                  fit_params={"estimator__bins": bins[l], "estimator__subsample": subsample_size},
                                  verbose=True, cv=cv, n_jobs=n_jobs)
        gridSearch.fit(X, y)

        # Output
        importance = pd.Series(e.feature_importances_, index=columns_)
        importance.sort(ascending=False)
        print "GS scores: {0}".format(gridSearch.grid_scores_)
        print "GS best params: {0}, score {1}".format(gridSearch.best_params_, gridSearch.best_score_)
        print "features {0}, train score {1}".format(gridSearch.best_estimator_.steps[0][-1].important_columns_.size, scorer(gridSearch.best_estimator_, X, y))
        print importance[:50]

        output["clust"][l] = {"dataTransformer": dataTransformer,
                              "model": gridSearch.best_estimator_,
                              "importance": importance
                              }
        pred[labels == l] = gridSearch.best_estimator_.predict_proba(X)

    # Score for all data
    prob = np.cumsum(pred[:, :-1], axis=1)
    print "total train score: ", calc_crps(prob, target.reshape((-1, 1)))
    cPickle.dump(output, open(gs_output, "w"))


def model_validation(datafile, gs_output, modelfile, subsample_size=50000, cv=6, n_jobs=2):
    """
        Performs k-fold cross-validation for model with best parameters (best params are loaded from @gs_output)

    :param datafile: str
        Path to a data file
    :param gs_output: str
        cPickle-filename with grid search output
    :param modelfile: str
        cPickle-filename to dump final model
    :param subsample_size: int
        subsample size to fit the model
    :param cv: int
        Number of folds in cross-validation
    :param n_jobs: int
        Number of jobs to run in parallel
    """

    # Load data
    data, Id, target, columns = load_data(datafile, test=False)

    best_output = cPickle.load(open(gs_output))

    model = MyPredictor(cluster={"cluster": best_output["clusterer"]},
                        columns=best_output["columns"],
                        dataTransformer={k: v["dataTransformer"] for k, v in best_output["clust"].iteritems()},
                        model={k: v["model"] for k, v in best_output["clust"].iteritems()},
                        base_predictor=BASE_ESTIMATOR,
                        bins={k: v for k, v in best_output["bins"].iteritems()},
                        subsample=subsample_size
                        )
    print data.shape
    gridSearch = GridSearchCV(model, param_grid={}, scoring=scorer, verbose=True, cv=cv, n_jobs=n_jobs)
    gridSearch.fit(data, target)
    print "GS scores: {0}".format(gridSearch.grid_scores_)
    print "train score {0}".format(scorer(gridSearch.best_estimator_, data, target))

    cPickle.dump(gridSearch.best_estimator_, open(modelfile, "w"))


def submission(datafile, modelfile, outputfile):

    model = cPickle.load(open(modelfile))

    test_data, Id, columns = load_data(datafile, test=True)

    pred = model.predict_proba(test_data)
    prob = np.cumsum(pred[:, :-1], axis=1)

    submission = pd.DataFrame(prob, columns=["Predicted" + str(i) for i in range(70)])
    submission["Id"] = Id.astype(int)
    submission = submission.reindex(columns=(["Id"] + ["Predicted" + str(i) for i in range(70)]))
    submission.to_csv(outputfile, index=False)


if __name__ == "__main__":

    grid_search(datafile="data/train_preprocessed.csv", gs_output="gs_output.pkl", subsample_size=100000, cv=3, n_jobs=3)
    model_validation(datafile="data/train_preprocessed.csv", gs_output="gs_output.pkl", modelfile="model.pkl", subsample_size=200000, cv=4, n_jobs=1)
    submission("data/test_preprocessed.csv", "model.pkl", "submission.csv")