__author__ = 'alfiya'
from model import *


def calcCRPS(predFile, expectedFile):
    expected = pd.read_csv(expectedFile, sep=" ", dtype=str, na_filter=False, header=None, usecols=[0]).values.astype(float)
    prob = getProb(predFile)
    H = (np.tile(np.arange(70), (expected.size, 1)) >= np.tile(expected, (1, 70))).astype(float)
    N = prob.shape[0]
    C = ((prob - H)**2).sum() / (70 * N)
    return C


def getProb(predFile, bias=2.45, outlier_value=25, scale=1):

    pred = np.loadtxt(predFile, dtype=float)
    pred = pred.reshape((pred.size, 1))

    prob = stats.logistic.cdf(x=np.tile(np.arange(70), (pred.size, 1)), loc=(np.tile(pred, (1, 70)) - bias), scale=scale)
    # if zero.any():
    #     prob[zero, :] = np.tile(np.repeat(1, 70), (zero.sum(), 1))
    # if outliers.any():
    #     # prob[outliers, :] = stats.logistic.cdf(x=np.tile(np.arange(70), (outliers.sum(), 1)), loc=(np.tile(outlier_value, (1, 70)) - bias))
    #     prob[outliers, :] = np.tile(np.repeat(0, 70), (outliers.sum(), 1))
    return prob


# class FMRegressor(BaseEstimator):
#     def __init__(self):
#         self.iter = iter_
#         self.stdev = stdev
#         self
#
#     def fit(self, X, y):
#         # ./libFM -task r -train train.libfm -test validation.libfm -meta libfmdata.meta -dim '1,1,8' -iter 10000 -method mcmc -init_stdev 0.1 -out prediction.txt


if __name__ == "__main__":

    print calcCRPS("data/prediction.txt", "data/validation.libfm")