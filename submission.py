__author__ = 'alfiya'
__author__ = 'alfiya'
from model import *
import datetime
CLASSIF = True

if CLASSIF:
    BASE_PREDICTOR = "GBCT_"
    my_predictor = "GBCT"
else:
    BASE_PREDICTOR = "GradientBoostingRegressor"
    my_predictor = "GBRT"


best_output = cPickle.load(open("best_classif_model.pkl"))
# Load data
data = pd.read_csv("data/train_preprocessed.csv")
Id = data["Id"].copy()
data.drop("Id", axis=1, inplace=True)
target = data["Expected"].copy().values
target[target > 70.] = 70.
data.drop("Expected", axis=1, inplace=True)
columns = data.columns.values
data = data.values

labels = best_output["clusterer"].transform(data)
print time.time()
pred = np.ones((data.shape[0], 71), dtype=np.float32)
for l in np.unique(labels):
    n = (labels == l).sum()
    dataTransformer = best_output["clust"][l]["dataTransformer"]
    X, columns_ = dataTransformer.transform(data[labels == l])
    print datetime.datetime.now()
    y = target[labels == l]

    pred[labels == l] = best_output["clust"][l]["model"].predict_proba(X)
prob = np.cumsum(pred[:, :-1], axis=1)
print "total score: ", calc_crps(prob, target.reshape((target.size, 1)))

#*********** Submission *************

start = time.time()
# MODEL VALIDATION
print datetime.datetime.now()
for k, v in best_output["clust"].iteritems():
    best_output["clust"][k]["model"].set_params(**{"regressor__n_estimators": 200})
    print best_output["clust"][k]["model"].get_params()
model = MyPredictor(cluster={"cluster": best_output["clusterer"]}, columns=best_output["columns"],
                    dataTransformer={k: v["dataTransformer"] for k, v in best_output["clust"].iteritems()},
                    model_params={k: v["model"].get_params() for k, v in best_output["clust"].iteritems()},
                    base_predictor=my_predictor,
                    bins={k: v for k, v in best_output["bins"].iteritems()},
                    subsample=500000
                    )
scorer_ = partial(scorer, classif=CLASSIF, has_labels=False,
                  weights={k: v["weights"] for k, v in best_output["bins"].iteritems()})
# gridSearch = GridSearchCV(model, param_grid={}, scoring=scorer_, verbose=True, cv=2, n_jobs=1)
# gridSearch.fit(data, target)
# print "GS scores: {0}".format(gridSearch.grid_scores_)


model.fit(data, target)
cPickle.dump(model, open("classif_model_200.pkl", "w"))
print "train score {0}, time {1}".format(scorer_(model, data, target), time_passed(start))


# print "total time {0}".format(time_passed(start_total))
# SUBMISSION
model = cPickle.load(open("classif_model_200.pkl"))
test_data = pd.read_csv("data/test_preprocessed.csv")

Id = test_data["Id"].copy()
test_data.drop("Id", axis=1, inplace=True)

test_data = test_data.values
pred = model.predict_proba(test_data)
# labels = gridSearch.best_estimator_.labels(test_data)
# pred = np.ones((test_data.shape[0], 71), dtype=np.float32)
# for l in np.unique(labels):
#     pred[labels == l] = get_prob_from_joined_classes(pred[l], weights=best_output["bins"][l]["weights"])
prob = np.cumsum(pred[:, :-1], axis=1)

# prob = getProb(pred=pred) #, outlier_value=best_outlier_value)

submission = pd.DataFrame(prob, columns=["Predicted"+str(i) for i in range(70)])
submission["Id"] = Id.astype(int)
submission = submission.reindex(columns=(["Id"]+["Predicted"+str(i) for i in range(70)]))
submission.to_csv("submission.csv", index=False)

