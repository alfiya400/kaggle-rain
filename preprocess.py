__author__ = 'alfiya'
import csv
import pandas as pd
import numpy as np
import time
import cPickle
from joblib import Parallel, delayed

NAN = np.array([-99900.0, np.nan, -99903.0, -99901.0, 999.0])  # values corresponding to nan

HYDROMETEOR_TYPE = {
    0: "no_echo",
    1: "moderate_rain",
    2: "moderate_rain",
    3: "heavy_rain",
    4: "rain/hail",
    5: "big_drops",
    6: "AP",
    7: "Birds",
    8: "unknown",
    9: "no_echo",
    10: "dry_snow",
    11: "wet_snow",
    12: "ice_crystals",
    13: "graupel",
    14: "graupel"}

PRECIP = HYDROMETEOR_TYPE.values()

STATS = [u'mean', u'std', u'min', u'50%', u'max']  # aggregations


def calc_aggregations(seq):
    """
    Used in "appendData" function
    function calculates aggregations for @seq
    Null values are ignored
    :param seq: numpy.array
    :return: list
        List of aggregations
    """
    not_null = (~pd.isnull(seq))
    seq_not_null = seq[not_null]
    if seq_not_null.size > 1:
        stat = [seq_not_null.mean(), seq_not_null.std(ddof=1), seq_not_null.min(),
                np.percentile(seq_not_null, 50), seq_not_null.max()]
    elif seq_not_null.size == 1:
        _ = seq_not_null[0]
        stat = [_, 0, _, _, _]
    else:
        stat = [np.nan] * 5
    return stat


def stats_per_feature(l, idx, h, columns, new_row):
    """
    Used in "transform" function
    Calculates aggregations for @l

    :param l: numpy.array
    :param idx: numpy.array
    :param h: str
        feature name (e.g. DistanceToRadar, TimeToEnd, RR1 etc.)
    :param columns: list
        new features names (e.g. DistanceToRadar_min, DistanceToRadar_max etc.)
    :param new_row: list
        new features values
    :return:
    """

    # For HydrometeorType calculate most frequent type and then transform it to a vector of all hydrometeor types
    # In this vector:
    #   values is 0 if type is not the most frequent type
    #   value equals to a frequency if type is most frequent
    if h == "HydrometeorType":

        p_map = {k: 0 for k in PRECIP}  # init vector of frequencies with 0

        # calc frequencies
        for v in l:
            p_map[HYDROMETEOR_TYPE[v]] += 1
        # most frequent type
        best = p_map.keys()[np.argmax(p_map.values())]

        # create final vector
        for p in PRECIP:
            columns.append(p)
            if p == best:
                new_row.append(p_map[p])
            else:
                new_row.append(0)

    elif h == "DistanceToRadar":
        values = l[idx]  # get only 1 value for each radar
        columns.extend([h + "_" + k for k in [u'min', u'50%', u'mean', u'max']])
        new_row.extend([values.min(), np.percentile(values, 50), values.mean(), values.max()])
    elif h == "TimeToEnd":
        columns.append("time")
        new_row.append((l.max() - l.min() + 6.) / 60.)  # total time
        columns.append("n_obs")
        new_row.append(l.size)  # total number of observations

    elif h == "Kdp":
        # calc KDP statistics with RR3
        pass
    elif h == "RR3":
        stat = calc_aggregations(l)
        columns.extend([h + "_" + k for k in STATS])
        new_row.extend(stat)

        # calc KDP
        kdp = np.sign(l) * np.exp(np.log(np.abs(l) / 40.6) / 0.866)  # got this formula from forum
        stat = calc_aggregations(kdp)
        columns.extend(["Kdp" + "_" + k for k in STATS])
        new_row.extend(stat)

    else:
        stat = calc_aggregations(l)
        columns.extend([h + "_" + k for k in STATS])
        new_row.extend(stat)


def get_values(s):
    """
    Used in "transform function"
    Extracts numeric values from string and replaces values specified in NAN to np.nan
    :param s: str
    :return: numpy.array
    """
    l = np.array(s.split(" "), dtype=float)
    nan = np.in1d(l, NAN)
    l[nan] = np.nan
    return l


def transform(row, test=False):
    """
    Used in "preprocess" function
    This function extracts features from the @row.
    For each observation from radars it calculated aggregations

    :param row: pandas.Series
        Contains list of observations from radars over 1 hour
    :param test:
        test file indicator
        if True, then supposed that there is no "Expected" feature
    :return:
    """
    # Extract Id
    new_row = [row["Id"]]
    columns = ["Id"]

    # Get staring indexes for each radar
    # New radar detected if "TimeToEnd" value increased or
    # if "DistanceToRadar" value changed (as far as radars can't move)
    l_timeToEnd = get_values(row["TimeToEnd"])

    # Indicator of increase in value TimeToEnd
    idx_timeToEnd = np.insert(np.where(np.diff(l_timeToEnd) > 0)[0], 0, 0)

    l_distance = get_values(row["DistanceToRadar"])
    # indicator of change in value DistanceToRadar
    idx = np.insert(np.where(np.diff(l_distance) != 0)[0], 0, 0)

    # final list of starting indices for radars
    idx_ = np.unique(np.concatenate([idx, idx_timeToEnd])) if idx.size > 1 else idx_timeToEnd

    # calc statistics for timeToEnd and DistanceToRadar
    stats_per_feature(l_timeToEnd, idx_, "TimeToEnd", columns, new_row)
    stats_per_feature(l_distance, idx_, "DistanceToRadar", columns, new_row)

    if test:
        drop_cols = ["Id", "TimeToEnd", "DistanceToRadar"]
    else:
        drop_cols = ["Id", "Expected", "TimeToEnd", "DistanceToRadar"]

    # calc statistics for any other features
    for h, v in row.drop(drop_cols).iteritems():
        l = get_values(v)
        stats_per_feature(l, idx_, h, columns, new_row)

    if test:
        columns.append("nRadars")
        new_row.append(idx_.size)
    else:
        columns.extend(["nRadars", "Expected"])
        new_row.extend([idx_.size, row["Expected"]])
    return pd.Series(new_row, columns)


def foo(x, transform, axis=1, test=False):
    """
    Supporting function, used only in "preprocess" function
    :param x:
    :param transform:
    :param axis:
    :param test:
    :return:
    """
    return x.apply(transform, axis=axis, args=(test,))


def preprocess(file_in, file_out, test=False, n_jobs=6):
    """
    This function preprocesses raw data file.
    For each row and for each feature it extracts aggregations over TimeToEnd:
        From feature TimeToEnd it extracts total time ("time") and number of observations ("n_obs")
        From feature DistanceToRadar it extracts aggregations ('min', '50% quantile', 'mean', 'max')
        For any other features it calculates ('mean', 'std', 'min', '50% quantile', 'max')

        New features names follow the pattern: <feature name>_<aggregation function>

    Parameters
    ----------
    :param file_in: str
        csv-file name for data to be preprocessed
    :param file_out: str
        csv-file name for output data
    :param test: bool
        indicator for test data (data without label)
    :return:
    """
    # Load data to pandas.DataFrame
    data_raw = pd.read_csv(file_in, na_filter=False, chunksize=5000)

    # Apply transformations to data chunks in parallel
    start = time.time()
    data = Parallel(n_jobs=n_jobs, verbose=11)(delayed(foo)(x, transform, axis=1, test=test) for i, x in enumerate(data_raw))
    print "Preprocessing time: ", round((time.time() - start) / 60, 3)
    print "Records: ", len(data)

    # Join data chunks and save result to csv
    data = pd.concat(data)
    data.to_csv(file_out, index=False)

    print "File", file_in, "preprocessed to", file_out


def corr_matrix(file_in, corr_file):
    """
    This function loads data from @file_in to pandas.DataFrame,
    calculates Pearson correlations between features and dumps result into @corr_file using cPickle

    :param file_in: str
        csv-file name to data
    :param corr_file: str
        filename for correlation matrix
    :return:
    """
    data = pd.read_csv(file_in)
    data.drop("Id", axis=1, inplace=True)

    corr = data.corr()
    with open(corr_file, "w") as f:
        cPickle.dump(corr, f)


if __name__ == "__main__":

    preprocess("data/train_2013.csv", "data/train_preprocessed.csv", n_jobs=6)
    preprocess("data/test_2014.csv", "data/test_preprocessed.csv", test=True, n_jobs=6)

    corr_matrix("data/train_preprocessed.csv", "data/corr_matrix.pkl")
