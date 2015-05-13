__author__ = 'alfiya'
from model import *

if __name__ == "__main__":
    CLASSIF = True

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
        bin_weights.pop()
        print bc
        print np.bincount(target_new[labels == l])
        print bin_weights
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
        print X.shape
        big_cluster = X.shape[0] > 500000

        start = time.time()
        e = globals()[my_predictor](max_depth=6 if big_cluster else 3, learning_rate=0.05, n_estimators=100, min_samples_leaf=10, random_state=0)
        e.fit(X, y, bins=bins[l]["bins"], subsample=400000)
        scorer_ = partial(scorer, classif=CLASSIF, weights=bins[l]["weights"])
        print "primary score {0}, time {1}".format(scorer_(e, X, y), timePassed(start))

        prediction_model = Pipeline([("featureSelector", FeatureSelector(e.feature_importances_, 0.004, columns_)),
                                    ("regressor", globals()[my_predictor](min_samples_leaf=20, learning_rate=0.05, n_estimators=100, subsample=0.8, random_state=0))
                                     ])
        param_grid = dict(featureSelector__importance_threshold=[0.0, 0.0001, 0.001, 0.01, 0.02, 0.03],
                          regressor__max_depth=[6 if big_cluster else 3])
        gridSearch = GridSearchCV(prediction_model, param_grid=param_grid, scoring=scorer_, fit_params={"regressor__bins": bins[l]["bins"]}, verbose=True, cv=6, n_jobs=2)
        gridSearch.fit(X, y)

        # output
        importance = pd.Series(gridSearch.best_estimator_.steps[-1][-1].feature_importances_, index=gridSearch.best_estimator_.steps[0][-1].important_columns_)
        importance.sort(ascending=False)
        print "GS scores: {0}".format(gridSearch.grid_scores_)
        print "GS best params: {0}, score {1}".format(gridSearch.best_params_, gridSearch.best_score_)
        print "features {0}, train score {1}".format(importance.size, scorer_(gridSearch.best_estimator_, X, y))
        print importance
        print confusion_matrix(y > 1, gridSearch.best_estimator_.predict(X) > 1)

        output["clust"][l] = {"dataTransformer": dataTransformer,
                              "model": gridSearch.best_estimator_,
                              "importance": importance,
                              "scores": gridSearch.grid_scores_,
                              "best_params": gridSearch.best_params_}
        pred[l] = gridSearch.best_estimator_.predict_proba(X)

    pred_ = np.ones((data.shape[0], 71), dtype=np.float32)
    for l in np.unique(labels):
        pred_[labels == l] = getPred(pred[l], weights=bins[l]["weights"])
    prob = np.cumsum(pred_[:, :-1], axis=1)
    print "total score: ", calcCRPS(prob, target.reshape((target.size, 1)))
    cPickle.dump(output, open("best_classif_model.pkl", "w"))

    #*********** Submission *************

    best_output = cPickle.load(open("best_classif_model.pkl"))

    # MODEL VALIDATION
    model = myPredictor(cluster={"cluster": best_output["clusterer"]}, columns=best_output["columns"],
                        dataTransformer={k: v["dataTransformer"] for k, v in best_output["clust"].iteritems()},
                        model_params={k: v["model"].get_params() for k, v in best_output["clust"].iteritems()},
                        base_predictor=my_predictor,
                        bins={k: v["bins"] for k, v in best_output["bins"].iteritems()})
    scorer_ = partial(scorer, classif=CLASSIF, has_labels=True,
                      weights={k: v["weights"] for k, v in best_output["bins"].iteritems()})
    gridSearch = GridSearchCV(model, param_grid={}, scoring=scorer_, verbose=True, cv=2, n_jobs=1)
    gridSearch.fit(data, target)
    print "GS scores: {0}".format(gridSearch.grid_scores_)
    print "train score {0}".format(scorer_(gridSearch.best_estimator_, data, target))

    # SUBMISSION
    test_data = pd.read_csv("data/test_preprocessed.csv")

    Id = test_data["Id"].copy()
    test_data.drop("Id", axis=1, inplace=True)

    test_data = test_data.values
    pred = gridSearch.best_estimator_.predict_proba(test_data)
    labels = gridSearch.best_estimator_.labels(test_data)
    pred_ = np.ones((test_data.shape[0], 71), dtype=np.float32)
    for l in np.unique(labels):
        pred_[labels == l] = getPred(pred[l], weights=best_output["bins"][l]["weights"])
    prob = np.cumsum(pred_[:, :-1], axis=1)

    # prob = getProb(pred=pred) #, outlier_value=best_outlier_value)

    submission = pd.DataFrame(prob, columns=["Predicted"+str(i) for i in range(70)])
    submission["Id"] = Id.astype(int)
    submission = submission.reindex(columns=(["Id"]+["Predicted"+str(i) for i in range(70)]))
    submission.to_csv("submission.csv", index=False)


