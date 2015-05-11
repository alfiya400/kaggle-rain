__author__ = 'alfiya'

from sklearn.tree import _tree
from sklearn.utils import array2d
from sklearn.base import BaseEstimator
from model import *


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
        y[y > self.outlier_threshold] = self.outlier_threshold
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

if __name__ == "__main__":
    data_transformer = DataTransformer()
    Id, data, target = data_transformer.fit_transform("data/train_preprocessed.csv")

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
    # with open("best_estimator.pkl", "w") as f:
    #     cPickle.dump(best_estimator, f)
    # importance.to_csv("importance.csv")

    # [mean: -0.00865, std: 0.00016, params: {'n_estimators': 30, 'subsample': 0.9, 'learning_rate': 0.1, 'max_depth': 6}]
    # {'n_estimators': 30, 'subsample': 0.9, 'learning_rate': 0.1, 'max_depth': 6} -0.00864635259266
    # -0.00848593007603