__author__ = 'alfiya'
from model import *

if __name__ == "__main__":
    start_total = time.time()
    CLASSIF = True

    if CLASSIF:
        BASE_PREDICTOR = "GBCT_"
        my_predictor = "GBCT"
    else:
        BASE_PREDICTOR = "GradientBoostingRegressor"
        my_predictor = "GBRT"

    # Load data
    data = pd.read_csv("data/train_preprocessed.csv")
    Id = data["Id"].copy()
    data.drop("Id", axis=1, inplace=True)
    target = data["Expected"].copy().values
    target[target > 70.] = 70.
    data.drop("Expected", axis=1, inplace=True)
    columns = data.columns.values
    data = data.values

    # Clustering
    clust = Clusterer(clusterer="KMeans", n_clusters=2, columns=columns)
    labels = clust.fit_transform(data)

    # Join small classes into bigger ones
    bins = {}
    target_int = np.ceil(target).astype(np.int8)
    target_new = np.zeros(target_int.shape, dtype=int)
    for l in np.unique(labels):
        n = (labels == l).sum()
        min_bin_count = 4000 if n > 500000 else 3000
        bc = np.bincount(target_int[labels == l])
        bins_, bin_count, bin_number = [], 0., 0
        bin_weights = [np.array([])]
        for v in bc:
            bin_count += v
            bins_.append(bin_number)
            bin_weights[bin_number] = np.append(bin_weights[bin_number], v)

            if bin_count > min_bin_count:
                bin_weights[bin_number] /= bin_count
                bin_count = 0
                bin_number += 1
                bin_weights.append(np.array([]))

        target_new[labels == l] = np.vectorize(lambda x: bins_[x])(target_int[labels == l])
        if len(bin_weights[-1]):
            bin_weights[-1] /= bin_count
        else:
            bin_weights.pop()
        # print bc
        print np.bincount(target_new[labels == l])
        # print bin_weights
        bins[l] = {"bins": bins_, "weights": bin_weights}

    # target_old = target.copy()
    # target = target_new.copy()
    # del target_new

    print "******Grid search******"
    output = {"clusterer": clust, "columns": columns, "clust": {}, "bins": bins}
    pred = {}
    for l in np.unique(labels):
        n = (labels == l).sum()
        dataTransformer = DataTransformer(columns=columns, corr_threshold=0.98)
        X, columns_ = dataTransformer.fit_transform(data[labels == l])
        y = target[labels == l]
        # y = np.ceil(y).astype(np.int8)
        # y = np.vectorize(lambda x: bins[l]["bins"][x])(y)
        print X.shape
        big_cluster = X.shape[0] > 500000

        start = time.time()
        predictor_params = dict(min_samples_leaf=20, random_state=0) #max_depth=6 if big_cluster else 3,  , learning_rate=0.05, n_estimators=10
        e = globals()[BASE_PREDICTOR](**predictor_params)  # globals()[my_predictor] **predictor_params
        e.fit(X, y, bins=bins[l], subsample=False)

        scorer_ = partial(scorer, classif=CLASSIF, weights=bins[l]["weights"])
        print "primary score {0}, time {1}".format(scorer_(e, X, y), time_passed(start))

        predictor_params = dict(min_child_weight=20, learning_rate=0.05, n_estimators=100, subsample=0.8)  # , nthread=3 if big_cluster else 5
        prediction_model = Pipeline([("featureSelector", FeatureSelector(e.feature_importances_, 0.004, columns_)),
                                    ("regressor", globals()[my_predictor](**predictor_params))
                                     ])
        param_grid = dict(featureSelector__importance_threshold=0.001,
                          regressor__max_depth=5 if big_cluster else 3,
                          regressor__subsample=0.8, regressor__n_estimators=150,
                          regressor__learning_rate=0.1)
        # gridSearch = GridSearchCV(prediction_model, param_grid=param_grid, scoring=scorer_, fit_params={"regressor__bins": bins[l], "regressor__subsample": 50000}, verbose=True, cv=3, n_jobs=3)
        # gridSearch.fit(X, y)
        start = time.time()
        prediction_model.set_params(**param_grid)
        prediction_model.fit(X, y, **{"regressor__bins": bins[l], "regressor__subsample": 50000})
        print "features {0}, train score {1}".format(prediction_model.steps[0][-1].important_columns_.size, scorer_(prediction_model, X, y))
        # output
        # importance = pd.Series(gridSearch.best_estimator_.steps[-1][-1].feature_importances_, index=gridSearch.best_estimator_.steps[0][-1].important_columns_)
        importance = pd.Series(e.feature_importances_, index=columns_)
        importance.sort(ascending=False)
        # print "GS scores: {0}".format(gridSearch.grid_scores_)
        # print "GS best params: {0}, score {1}".format(gridSearch.best_params_, gridSearch.best_score_)
        # print "features {0}, train score {1}".format(gridSearch.best_estimator_.steps[0][-1].important_columns_.size, scorer_(gridSearch.best_estimator_, X, y))
        print importance[:50]
        # print confusion_matrix(y > 1, gridSearch.best_estimator_.predict(X) > 1)

        output["clust"][l] = {"dataTransformer": dataTransformer,
                              "model": prediction_model,  # gridSearch.best_estimator_,
                              "importance": importance,
                              "scores": None,  # gridSearch.grid_scores_,
                              "best_params": None}  # gridSearch.best_params_}
        pred[l] = prediction_model.predict_proba(X)  # gridSearch.best_estimator_.predict_proba(X)
        print "time {0}".format(time_passed(start))
    pred_ = np.ones((data.shape[0], 71), dtype=np.float32)
    for l in np.unique(labels):
        pred_[labels == l] = pred[l]
    prob = np.cumsum(pred_[:, :-1], axis=1)
    print "total score: ", calc_crps(prob, target.reshape((target.size, 1)))
    cPickle.dump(output, open("best_classif_model.pkl", "w"))

    #*********** Submission *************

    best_output = cPickle.load(open("best_classif_model.pkl"))

    # MODEL VALIDATION
    model = MyPredictor(cluster={"cluster": best_output["clusterer"]}, columns=best_output["columns"],
                        dataTransformer={k: v["dataTransformer"] for k, v in best_output["clust"].iteritems()},
                        model_params={k: v["model"].get_params() for k, v in best_output["clust"].iteritems()},
                        base_predictor=my_predictor,
                        bins={k: v for k, v in best_output["bins"].iteritems()},
                        subsample=50000
                        )
    scorer_ = partial(scorer, classif=CLASSIF, has_labels=False,
                      weights={k: v["weights"] for k, v in best_output["bins"].iteritems()})
    gridSearch = GridSearchCV(model, param_grid={}, scoring=scorer_, verbose=True, cv=6, n_jobs=1)
    gridSearch.fit(data, target)
    print "GS scores: {0}".format(gridSearch.grid_scores_)
    print "train score {0}".format(scorer_(gridSearch.best_estimator_, data, target))

    print "total time {0}".format(time_passed(start_total))
    # SUBMISSION
    test_data = pd.read_csv("data/test_preprocessed.csv")

    Id = test_data["Id"].copy()
    test_data.drop("Id", axis=1, inplace=True)

    test_data = test_data.values
    pred = gridSearch.best_estimator_.predict_proba(test_data)
    labels = gridSearch.best_estimator_.labels(test_data)
    pred_ = np.ones((test_data.shape[0], 71), dtype=np.float32)
    for l in np.unique(labels):
        pred_[labels == l] = get_prob_from_joined_classes(pred[l], weights=best_output["bins"][l]["weights"])
    prob = np.cumsum(pred_[:, :-1], axis=1)

    # prob = getProb(pred=pred) #, outlier_value=best_outlier_value)

    submission = pd.DataFrame(prob, columns=["Predicted"+str(i) for i in range(70)])
    submission["Id"] = Id.astype(int)
    submission = submission.reindex(columns=(["Id"]+["Predicted"+str(i) for i in range(70)]))
    submission.to_csv("submission.csv", index=False)


