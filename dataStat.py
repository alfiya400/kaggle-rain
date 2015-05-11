    __author__ = 'alfiya'
    from model import *


    def cluster_stats(data, y, labels, filename, threshold=69):
        from pandas.tools.plotting import table
        import matplotlib.gridspec as gridspec
        target = (y > threshold).astype(int)
        describe = pd.Series(y).groupby(labels).describe()
        score = adjusted_rand_score(labels, target)
        subplot_width = 7
        subplot_height = 4
        fig = plt.figure(figsize=(subplot_width*2, subplot_height*5))
        gs = gridspec.GridSpec(5, 2, height_ratios=[1, 3, 3, 3, 3])

        ax = plt.subplot(gs[0, :])
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_title("Describe by cluster, ARS {0:.5f}".format(score))
        tbl = table(ax, np.round(describe, 4).unstack(), loc="center", colWidths=[0.1]*8)
        tbl.set_fontsize(24)
        tbl.scale(1.2, 1.2)

        randomState = np.random.RandomState(15)
        idx = randomState.choice(a=np.where(labels > -1)[0], size=15000, replace=False)
        y0 = randomState.choice(a=np.where(labels == 0)[0], size=7000, replace=False)
        y1 = randomState.choice(a=np.where(labels == 1)[0], size=7000, replace=False)

        tmp = data.iloc[idx, :].copy()
        tmp["Expected"] = target[idx]

        ax = plt.subplot(gs[1, 0])
        radviz(tmp, "Expected", colormap='winter', ax=ax)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        ax = plt.subplot(gs[1, 1])
        radviz(data.iloc[idx, :], "Cluster", colormap='winter', ax=ax)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        ax = plt.subplot(gs[2, 0])
        ax = andrews_curves(tmp[tmp["Expected"].values == 0], "Expected", colormap='winter', ax=ax)
        x_lim = (-3, 3)
        y_lim = ax.get_ylim()
        ax.set_ylim(y_lim)
        ax.set_xlim(x_lim)
        ax.grid(True)

        ax = plt.subplot(gs[2, 1])
        ax = andrews_curves(tmp[tmp["Expected"].values == 1], "Expected", colormap='winter', ax=ax)
        ax.set_ylim(y_lim)
        ax.set_xlim(x_lim)
        ax.grid(True)

        ax = plt.subplot(gs[3, 0])
        ax = andrews_curves(data.iloc[y0,], "Cluster", colormap='winter', ax=ax)
        ax.set_ylim(y_lim)
        ax.set_xlim(x_lim)
        ax.grid(True)

        ax = plt.subplot(gs[3, 1])
        ax = andrews_curves(data.iloc[y1,], "Cluster", colormap='winter', ax=ax)
        ax.set_ylim(y_lim)
        ax.set_xlim(x_lim)
        ax.grid(True)

        ax = plt.subplot(gs[4, :])
        for c in np.unique(labels):
            sns.kdeplot(y[labels == c], ax=ax, label="Cluster {}".format(c), bw=0.4)
            # sns.kdeplot(y[data["Cluster"].values == 1], ax=ax, label="Cluster 1", bw=0.4)
        fig.tight_layout()

        plt.savefig(filename, format="png")


if __name__ == "__main__":
    data_transformer = DataTransformer(corr_threshold=0.95)
    Id, data, target = data_transformer.fit_transform("data/train_preprocessed.csv")

    best_estimator = cPickle.load(open("best_estimator.pkl"))

    Id, test_data = data_transformer.transform("data/test_preprocessed.csv")

    label = best_estimator.labels(data)
    pred = best_estimator.predict(data)

    cluster_stats(data, pred, label, "plots/train_pred_stat.png")
    cluster_stats(data, target.values, label, "plots/train_pred_stat.png")


    label = best_estimator.labels(data)
    pred = best_estimator.predict(data)
    cluster_stats(test_data, pred, label, "plots/test_pred_stat.png")




