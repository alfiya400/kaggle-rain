__author__ = 'alfiya'

from dataStat import *
from sklearn.base import BaseEstimator, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble.gradient_boosting import LogOddsEstimator
from sklearn.ensemble.gradient_boosting import BinomialDeviance
from sklearn.ensemble.gradient_boosting import PriorProbabilityEstimator
from sklearn.ensemble.gradient_boosting import MultinomialDeviance
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
sns.set()


class GBClassifier(BaseEstimator):
    def __init__(self,
                 stage_estimator=DecisionTreeRegressor,
                 stage_estimator_params={"max_depth": 6, "min_samples_leaf": 20},
                 n_estimators=15,
                 learning_rate=0.1,
                 init_estimator=LogOddsEstimator,
                 seed=0):
        self.stage_estimator = stage_estimator
        self.stage_estimator_params = stage_estimator_params
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = []
        self.init_estimator = init_estimator()
        self.randomState = np.random.RandomState(seed)

    def _stage_X_y(self, X, y):
        y0 = self.randomState.choice(a=np.where(y == 0)[0], size=y.sum(), replace=False)
        y1 = self.randomState.choice(a=np.where(y == 1)[0], size=y.sum(), replace=True)
        idx = np.concatenate([y0, y1])
        stage_X = X[idx, :]  # np.concatenate((X[y1, :], X[y0, :]), axis=0)
        stage_Y = y[idx]  # np.concatenate((y[y1, :], y[y0, :]))
        return stage_X, stage_Y, idx

    def fit(self, X, y):
        self.init_estimator.fit(X, y)
        prediction = self.init_estimator.predict(X)
        self.estimators.append(self.init_estimator)

        for stage in np.arange(n_estimators):
            stage_X, stage_Y, idx = self._stage_X_y(X, y)
            stage_pred = prediction[idx]
            residual = stage_Y - 1.0 / (1.0 + np.exp(-stage_pred))

            estimator = self.stage_estimator(**self.stage_estimator_params)
            estimator.fit(stage_X, residual)

            self.estimators.append(estimator)
            prediction += self.learning_rate * estimator.predict(X)
        return self

    def predict_class(self, X):
        prediction = self.init_estimator.predict(X)
        for stage in np.arange(n_estimators):
            prediction += self.learning_rate * self.estimators[stage].predict(X)

        return 1.0 / (1.0 + np.exp(-prediction))


class RFClassifier(BaseEstimator):
    def __init__(self, stage_estimator=None, stage_estimator_params=None, n_estimators=15, seed=0):
        if not stage_estimator_params:
            self.stage_estimator_params = {"max_depth": 6, "min_samples_leaf": 20}
        else:
            self.stage_estimator_params = stage_estimator_params
        if stage_estimator is None:
            self.stage_estimator = DecisionTreeClassifier(max_depth=6, min_samples_leaf=20)
        else:
            self.stage_estimator = stage_estimator

        self.n_estimators = n_estimators
        self.estimators = []
        self.randomState = np.random.RandomState(seed)
        self.n_features = None

    def _stage_X_y(self, X, y):
        y0 = self.randomState.choice(a=np.where(y == 0)[0], size=2*y.sum(), replace=False)
        y1 = self.randomState.choice(a=np.where(y == 1)[0], size=y.sum(), replace=True)
        idx = np.concatenate([y0, y1])
        stage_X = X[idx, :]
        stage_Y = y[idx]
        return stage_X, stage_Y, idx

    def fit(self, X, y):
        prediction = np.zeros(y.size)
        for stage in np.arange(self.n_estimators):
            stage_X, stage_Y, idx = self._stage_X_y(X, y)
            estimator = clone(self.stage_estimator)
            estimator.fit(stage_X, stage_Y)

            self.estimators.append(estimator)
            prediction += estimator.predict_proba(X)[:, 1]

        prediction /= self.n_estimators
        self.n_features = X.shape[1]
        return self

    def predict(self, X):
        prediction = np.zeros(X.shape[0])
        for stage in np.arange(self.n_estimators):
            prediction += self.estimators[stage].predict_proba(X)[:, 1]
        prediction /= self.n_estimators

        return prediction > 0.5

    def predict_proba(self, X):
        prediction = np.zeros(X.shape[0])
        for stage in np.arange(self.n_estimators):
            prediction += self.estimators[stage].predict_proba(X)[:, 1]
        prediction /= self.n_estimators

        return prediction

    @property
    def feature_importances_(self):
        return sum(tree.feature_importances_
                   for tree in self.estimators) / self.n_estimators


class classifier(GradientBoostingClassifier):
    def fit(self, X, y, monitor=None):
        y0 = np.where(y == 0)[0]
        y1 = randomState.choice(a=np.where(y == 1)[0], size=(y.size-y.sum())/2, replace=True)
        idx = np.concatenate([y0, y1])
        return super(classifier, self).fit(X[idx, :], y[idx], monitor)

if __name__ == "__main__":
    data_transformer = DataTransformer()
    Id, data, target = data_transformer.fit_transform("data/train_preprocessed.csv")

    threshold = 60.0
    outliers = target.values > threshold
    not_zero = (target.values > 0) & (target.values <= threshold)
    target[outliers] = 2
    target[not_zero] = 1
    outliersNum = outliers.sum()
    nonZeroNum = not_zero.sum()
    zeroNum = target.size - outliersNum - nonZeroNum

    print "******Grid search******"
    start = time.time()
    # estimator = classifier(max_depth=6, min_samples_leaf=20, learning_rate=0.05, n_estimators=40, subsample=0.8, random_state=0)
    # estimator = RFClassifier(n_estimators=100)
    estimator = RandomForestClassifier(max_depth=6, min_samples_leaf=20, n_estimators=10)
    sample_weight = np.ones(target.shape, dtype=int)
    weight1 = round(0.5 * zeroNum / nonZeroNum)
    weight2 = round(0.5 * zeroNum / outliersNum)
    sample_weight[not_zero] = weight1
    sample_weight[outliers] = weight2
    print weight1, weight2, np.unique(sample_weight)

    randomState = np.random.RandomState(0)
    gridSearch = GridSearchCV(estimator, param_grid={}, scoring=classification_scorer, verbose=True, cv=6, n_jobs=2, fit_params={"sample_weight": sample_weight})
    gridSearch.fit(data.values, target.values)
    scores = gridSearch.grid_scores_

    columns = list(data.columns) # + ["outlier"]
    importance = pd.Series(gridSearch.best_estimator_.feature_importances_, index=columns)
    importance.sort(ascending=False)
    print "time", round((time.time() - start)/60, 1)
    print scores
    print gridSearch.best_params_, gridSearch.best_score_
    print classification_scorer(gridSearch.best_estimator_, data, target)
    # loss = BinomialDeviance(2)
    # print loss(target, gridSearch.best_estimator_.predict_proba(data)[:, 1])
    print importance
    print confusion_matrix(target, gridSearch.best_estimator_.predict(data))
    cPickle.dump(gridSearch.best_estimator_, open("best_outliers_classifier.pkl", "w"))


    # subsample
    # idx = np.random.choice(data.index.values, 100, replace=False)
    #
    # plt_data = data.iloc[:, :60].loc[idx, :]
    # plt_data["class"] = target[idx]
    # print np.bincount(plt_data["class"].astype(int))
    #
    # sns.pairplot(plt_data, hue="class", size=2.5)
    # plt.savefig("pairwise_plot.pdf", format="pdf")

    #[[927404 193684]
 #[   771   4835]]
