# kaggle-rain

## About
Solution for kaggle [How Much Did It Rain](https://www.kaggle.com/c/how-much-did-it-rain) competition , score about 0.00775 on leaderboard.

To reproduce results first run "preprocess.py", then "classif_model.py"

#### Raw data transformation
  Every sample from raw data provides sequence of radar's measurements during the time, so for every sample and each measurement I calculate some aggregations (mean, min, max, 50% percentile, std) over the time.
  This step is implemented in [preprocessing.py](https://github.com/alfiya400/kaggle-rain/blob/master/preprocessing.py)

#### Primary analysis

You could find some primary analysis for preprocessed data in [primary_analysis.ipynb](https://github.com/alfiya400/kaggle-rain/blob/master/primary_analysis.ipynb)

#### Model
This step is implemented in [classif_model.py](https://github.com/alfiya400/kaggle-rain/blob/master/classif_model.py) 
##### Intuition
For model evaluation a [Continuous Ranked Probability Score](https://www.kaggle.com/c/how-much-did-it-rain/details/evaluation) is used, according to which I need to predict a cumulative probabilities for rain volume from 0 to 69.   

I consider it as a multi-class classification problem.
To build cumulative probabilities first I need predict a probability for each of the following rain gauge ranges: 0, (0, 1], (1, 2], ..., (68, 69], (69, +inf). In total there are 71 ranges, so this is a classification task with 71 classes.  

Some classes are very small (frequency < 500 are even smaller), so I join small classes into bigger bins, and use this bins as labels for classification. Then to get probabilities for each class I just distribute the probability of each bin between classes proportional to their frequencies (so bigger class will get bigger probability)

##### Details

- Clustering (`class Clusterer`).  
   I split the data into 2 clusters, then I fit model for each cluster. I've chosen 2 clusters because I saw 2 different patterns on Andrews Curves and distribution of rain gauge is different between clusters (see [primary_analysis.ipynb](https://github.com/alfiya400/kaggle-rain/blob/master/primary_analysis.ipynb))  
As a result I get one big cluster which contains about 80% of the data and small cluster with the rest 20%.

- Transformation of predicted variable (implemented in `grid\_search` function using function `get_bins`).  
   Transform the rain gauge (column "Expected") to 71 classes first and then join classes into bins. For big cluster I chose 4000 as minimum bin size (ends up in 6 bins), and 3000 - for the other cluster (end up in 10 bins)

- Preprocessing (`class DataTransformer`) & feature selection (`class FeatureImportanceEstimator and class FeatureSelector`).   
   During preprocessing I fill nan and inf with constant values, remove highly correlated features (with correlation > 98%), remove features with nan rate > 90%, add some intersections between features.
For feature selection I use feature importances from decision tree classifier (`class FeatureImportanceEstimator`) and I remove features with importance smaller than 0.001 (`class FeatureSelector`)

- Classification (`class GBCT`).  
   For classification I use [xgboost](https://github.com/dmlc/xgboost) implementation for gradient boosted machines. I train the model on a subsample of data (200 000 samples) to reduce the time required to fit the model

#### Hyperparameters grid search (`function grid_search`)
During grid search feature importances threshold and classificator's parameters can be fitted  

#### Final model cross-validation (`function model_validation`)
All steps (clustering, preprocessing & feature selection, classification) are joined in `class MyPredictor`, then K-fold is performed to evaluate the model.

I used 0.001 as feature importances threshold  
Classifier params:  
`n_estimators=100, learning_rate=0.1, subsample=100, min_child_weight=20, max_depth = 6 if big_cluster else 3`

My CV-score was about 0.0075, which is much better than the leaderboard score 0.0077, so I think my model has an overfitting problem.

#### Potential improvements for model

- Some hyperparameters are not included in grid search step (e.g. correlation threshold, high nan rate, intersections, bins size...) I set values for these parameters myself, but I should have chosen them using grid search 
- Fit several models on different subsamples of data and then build an ensemble
- Try different classifiers and then build an ensemble
- Check whether clustering is really nesessary
- More feature engineering
- Consider the problem as regression, then approximate cumulative probabilities with logistic CDF (itself it works worse than classification, I had about 0.0084 score on leaderboard). Then build a hybrid of regression and classification models 
