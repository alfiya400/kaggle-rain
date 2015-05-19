# kaggle-rain

## About
Solution for kaggle [How Much Did It Rain](https://www.kaggle.com/c/how-much-did-it-rain) competition , score about 0.00775 on leaderboard. 

#### Raw data transformation
  Every sample from raw data provides sequence of radar's measurements during the time, so for every sample I calculate from aggregation statistics over the time for each measurement.
  This step is implemented in [preprocessing.py](https://github.com/alfiya400/kaggle-rain/blob/master/preprocessing.py)

#### Primary analysis

You could find some primary analysis for preprocessed data in [primary_analysis.ipynb](https://github.com/alfiya400/kaggle-rain/blob/master/primary_analysis.ipynb)

#### Model
This step is implemented in [classif_model.py](https://github.com/alfiya400/kaggle-rain/blob/master/classif_model.py)  
For model evaluation a [Continuous Ranked Probability Score](https://www.kaggle.com/c/how-much-did-it-rain/details/evaluation) is used, according to which I need to predict a cumulative probabilities for rain volume from 0 to 69.   

To predict cumulative probabilities I want to predict a probability for each of the following rain volume ranges: 0, (0, 1], (1, 2], ..., (68, 69], (69, +inf). In total there are 71 ranges, so this is a classification task with 71 classes.  

Some classes are very small (frequency < 500), so I join small classes into bigger bins, and use this bins as labels for classification. Then to get probabilities for each class I just distribute the probability of each bin between classes proportional to their frequencies (so bigger class will get bigger probability)

Model consists of several steps:
- Clustering (class Clusterer).  
   I split the data into 2 clusters, then I analyse this clusters separately

- Transformation of predicted variable (function get_bins).  
   Transform the "Expected" column to 71 classes first and then join classes into bins

- Preprocessing (class DataTransformer) & feature selection (class FeatureImportanceEstimator and class FeatureSelector).   
   During preprocessing I fill nan and inf with constant values, remove highly correlated features (with correlation > 98%), remove features with na rate > 90%, add some intersections between features.
For feature selection I use feature importances build from decision tree (class FeatureImportanceEstimator), I remove features with importance smaller than 0.001 (class FeatureSelector)

- Classification (class GBCT).  
   For classification I use gradient boosted classifier from xgboost tool. I train the model on a subsample of data to reduce the time, required to fit the model

Finally I'd like to point out that all these steps are joined in class MyPredictor, which is made only to evaluate the final model using K-fold, when all hyperparameters search is done


## Reproduce results
To reproduce results first run "preprocess.py", then "classif_model.py"

