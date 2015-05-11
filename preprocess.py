__author__ = 'alfiya'
import csv
import pandas as pd
import numpy as np
import time
import cPickle
from joblib import Parallel, delayed
from charts.correlation_plots import correlation_plot

NAN = np.array([-99900.0, np.nan, -99903.0, -99901.0, 999.0])


def calcStat(seq):
    # u'mean', u'std', u'min', u'50%', u'max'
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

PRECIP = list(set(HYDROMETEOR_TYPE.values()))

STATS = [u'mean', u'std', u'min', u'50%', u'max']


def appendData(l, idx, h, columns, new_row, l_distance=None):

    # if h not in ("DistanceToRadar", "TimeToEnd"):
    #     min_distance_idx = np.where(l_distance == l_distance.min())[0]
    #     l = l[min_distance_idx]

    if h == "HydrometeorType":

        p_map = {k: 0 for k in PRECIP}
        for v in l:
            p_map[HYDROMETEOR_TYPE[v]] += 1
        best = p_map.keys()[np.argmax(p_map.values())]
        for p in PRECIP:
            columns.append(p)
            if p == best:
                new_row.append(p_map[p])
            else:
                new_row.append(0)
        # closest_distance_idx = idx[np.argmin(l_distance[idx])]
        # HydrometeorType_closest = HYDROMETEOR_TYPE[l[closest_distance_idx]]
        # for p in PRECIP:
        #     columns.append(p+"_closest")
        #     if p == HydrometeorType_closest:
        #         new_row.append(1) #p_map[p]
        #     else:
        #         new_row.append(0)

    elif h == "DistanceToRadar":
        values = l[idx]
        columns.extend([h + "_" + k for k in [u'min', u'50%', u'mean', u'max']])
        new_row.extend([values.min(), np.percentile(values, 50), values.mean(), values.max()])
    elif h == "TimeToEnd":
        columns.append("time")
        new_row.append((l.max() - l.min() + 6.) / 60.)
        columns.append("n_obs")
        new_row.append(l.size)

    elif h == "Kdp":
        pass
    elif h == "RR3":
        stat = calcStat(l)
        columns.extend([h + "_" + k for k in STATS])
        new_row.extend(stat)

        #calc KDP
        kdp = np.sign(l) * np.exp(np.log(np.abs(l)/40.6)/0.866)
        stat = calcStat(kdp)
        columns.extend(["Kdp" + "_" + k for k in STATS])
        new_row.extend(stat)

    else:
        stat = calcStat(l)
        columns.extend([h + "_" + k for k in STATS])
        new_row.extend(stat)


def getValues(s):
    l = np.array(s.split(" "), dtype=float)
    nan = np.in1d(l, NAN)
    l[nan] = np.nan
    return l


def transform(row, test=False):
    # row - Series
    new_row = [row["Id"]]
    columns = ["Id"]
    # get radars
    l_timeToEnd = getValues(row["TimeToEnd"])
    idx_timeToEnd = np.insert(np.where(np.diff(l_timeToEnd) > 0)[0], 0, 0)

    l_distance = getValues(row["DistanceToRadar"])
    idx = np.insert(np.where(np.diff(l_distance) != 0)[0], 0, 0)

    idx_ = np.unique(np.concatenate([idx, idx_timeToEnd])) if idx.size > 1 else idx_timeToEnd

    # append
    appendData(l_timeToEnd, idx_, "TimeToEnd", columns, new_row)
    appendData(l_distance, idx_, "DistanceToRadar", columns, new_row)

    if test:
        drop_cols = ["Id", "TimeToEnd", "DistanceToRadar"]
    else:
        drop_cols = ["Id", "Expected", "TimeToEnd", "DistanceToRadar"]

    for h, v in row.drop(drop_cols).iteritems():
        l = getValues(v)
        appendData(l, idx_, h, columns, new_row, l_distance)

    if test:
        columns.append("nRadars")
        new_row.append(idx_.size)
    else:
        columns.extend(["nRadars", "Expected"])
        new_row.extend([idx_.size, row["Expected"]])
    return pd.Series(new_row, columns)


def foo(x, transform, axis=1, test=False):
        return x.apply(transform, axis=axis, args=(test,))


def preprocess(file_in, file_out, test=False):

    data_raw = pd.read_csv(file_in, na_filter=False, chunksize=5000)
    start = time.time()
    # foo(data_raw.get_chunk(10), transform, axis=1)
    data = Parallel(n_jobs=6, verbose=11)(delayed(foo)(x, transform, axis=1, test=test) for i, x in enumerate(data_raw))
    print round((time.time() - start)/60, 3)
    print len(data)
    start = time.time()
    data = pd.concat(data)
    data.to_csv(file_out, index=False)
    print round((time.time() - start)/60, 3)

    print "File", file_in, "preprocessed to", file_out


def corr_matrix(file_in, corr_file):
    data = pd.read_csv(file_in)
    data.drop("Id", axis=1, inplace=True)

    corr = data.corr()
    with open(corr_file, "w") as f:
        cPickle.dump(corr, f)
    correlation_plot(corr, "plots/correlations.html")


def csv2libFM_row(x, y_name="Expected"):

    data_iter = ((str(i + 1), v) for i, v in enumerate(x.values if y_name is None else x.drop(y_name).values) if v)  # skip NA values
    y = min(70, float("0." if y_name is None else x[y_name]))

    return "{y} {data}".format(y=str(y),
                               data=" ".join([":".join(_) for _ in data_iter]))


def libfm_foo(x, transform, drop_cols, axis=1, y_name=None):
    x.drop(drop_cols, axis=1, inplace=True)
    return x.apply(transform, axis=axis, args=(y_name,))


def csv2libFM(filename, y_name=None, drop_cols=None):
    if drop_cols is None:
        drop_cols = ["Id"]
    data_raw = pd.read_csv(filename, na_filter=False, dtype=str, chunksize=10000)
    # data_raw.drop(drop_cols, axis=1, inplace=True)

    start = time.time()
    data = Parallel(n_jobs=4, verbose=11)(delayed(libfm_foo)(x, csv2libFM_row, y_name=y_name, drop_cols=drop_cols) for x in data_raw)
    data = pd.concat(data)

    file_out = ".".join([filename.split(".")[0], "libfm"])
    data.to_csv(file_out, index=False)

    t = round((time.time() - start) / 60, 3)

    print "File {filename} preprocessed to {file_out}, time {time}".format(filename=filename, file_out=file_out, time=t)


def meta_file(filename, drop_cols=None):
    if drop_cols is None:
        drop_cols = ["Expected", "Id"]
    data_raw = pd.read_csv(filename, dtype=float)
    data_raw.drop(drop_cols, axis=1, inplace=True)

    null_rate = np.round(pd.isnull(data_raw).sum() / float(data_raw.shape[0]), 5)
    uni = null_rate.unique()

    zero_iter = uni.size - 1
    zero_map = dict()
    for _ in null_rate[null_rate == 0].index.values:
        k = "PRECIP" if _ in PRECIP else _.split("_")[0]

        if k not in zero_map:
            zero_map[k] = [_]
        else:
            zero_map[k].append(_)

    inverted_zero_map = dict()
    for v in zero_map.itervalues():
        for v_ in v:
            inverted_zero_map[v_] = zero_iter
        zero_iter += 1

    groups = []
    for _ in data_raw.columns.values:
        if null_rate[_]:
            groups.append(str(np.where(uni == null_rate[_])[0][0] - 1) + "\n")
        else:
            groups.append(str(inverted_zero_map[_]) + "\n")

    with open("data/libfmdata.meta", "w") as f:
        f.writelines(groups)


if __name__ == "__main__":

    # preprocess("data/train_2013.csv", "data/train_preprocessed.csv")
    # preprocess("data/test_2014.csv", "data/test_preprocessed.csv", test=True)
    #
    # corr_matrix("data/train_preprocessed.csv", "data/corr_matrix.pkl")

    csv2libFM("data/train_preprocessed.csv", y_name="Expected")
    csv2libFM("data/test_preprocessed.csv", y_name=None)
    meta_file("data/train_preprocessed.csv", drop_cols=None)