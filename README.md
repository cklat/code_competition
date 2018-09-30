# IT-Talents Code Competition

This is my solution for September's 2018 code competition of IT-Talents.de

The aim of this competition was to predict the severity of injuries of accidents happening on (presumably) german roads. 

# Content Structure

The content of this repository is structured as the following:

- ##### ./submission.csv: This is the final submission file to the code competition, i.e. the best validated model executed on the provided *verkehrsunfaelle_test.csv* file

- ##### ./code/EDA.ipynb: This is a jupyter notebook prepared to do an exploratory analysis of the provided data. This notebook is intended for getting familiar with the data, detecting inconsistencies, check for missing values and maybe even acquire some insights for the Feature-Engineering process 

# General Approach

The data was fairly small and to some extent well documented. There were not missing values in either of the features, other than categorical values indicating that the exact value for a certain feature was unknown for a particular accident.

Also the split of training/test data is fairly generous, as we only have 1000 rows in the test data in contrast to ~15k in the training set which leaves us with many examples to train on. Moreover, the target variable seems to be equally distributed in the training and test set.

Regarding these underlying project and data conditions, I went for a light-weight approach in the form of a multi-class Classifier. Although, extra points will be awarded for Deep-Learning Frameworks, I find this to be an overkill for this competition as the provided is too small to train a Neural Network or similar approaches from scratch. 

## Pre-Processing/Feature-Engineering

Since I wasn't able to contribute so much time into this code competition, my pre-processing and feature-engineering step was reduced to the very essential steps. This became a step for cleaning up the raw data for the training phase rather than generating new and predictive features. Basically my contributions at this point can be summed up as the following:

- Delete/Rename Columns/Rows according to the findings in the EDA notebook

- Convert the provided date and time columns to a consent representation (see also the EDA notebook). Since the recorded weren't fully documented for a few rows (the year was missing), I reduced the date representation to a month and day representation, thus dropping the year information if given. If this wasn't the case and every row had the year documented, further (external) data could be introduced into the dataset. E.g. precise weather or calendar data.

  For the time column, I made the strict assumption that the exact time of the accidents weren't necessary, thus I dropped     the minute information of the time column. However, I haven't tested the impact on the performance if minutes were kept as   information for the classifier.

- Categorical columns were all one-hot-encoded, as this is normally a standard practice to deal with categorical columns. Furthermore, the converted date and time columns were also one-hot-encoded (e.g. 12 new columns for the months etc.) so that this information can be made processable for the classifier. There might be better options available for the latter part, however, due to time restrictions, I wasn't able to look into that further.

- Other than these steps, no further columns were introduced, as this step needs to completed with an even more intense data exploration. Since there are not many features given in the raw dataset and some of them seem to capture the same information (e.g. "Wetterlage" <-> "Bodenbeschaffenheit" or "Lichtverhältnisse <-> Zeit (24h)") it is not very trivial to come with some ideas on how to generate or link these given information in order to generate new features. As mentioned before, if the year was documented exhaustively, external date-related data could be introduced.

## Model Training

The choice of the model was fairly simple, as I have been working with the XGBoost Library before and it is probably the #1 tool to go in application like these. There are of course some other libraries (LightGBM, CatBoost) which could also be working pretty well on this dataset but I didn't have the time to implement a comparison part of model performance across libraries.

The following aspects were considered for training:

- The model is the XGBClassifier with a multi-class softmax objective, since we have 3 classes for the target variable
- Unfortunately, no further information was given regarding the metric that will be chosen for the evaluation. Also, no information was given, whether the classes will be weighted equally or not. Therefore, I assumed it will be a simple accuracy metric. However, the accuracy metric can be a very misleading metric, thus another metric like a multi-class, macro F1-Score or maybe even (if values in the target variable will be treated ordinary in the practical application) a RMSE metric.

- As for parameter tuning of the XGBoost model, I basically followed this tutorial (with minor changes): https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

- As the dataset was heavily imbalanced, it is hard to predict the '2' and '3' cases of the accident severity. There are probably ways to handle this scenario (perhaps generating synthetic data) but I wasn't able to look into that. However, we can set a naive benchmark for the model and the training dataset which is simply predicting the class '1' for the accident severity.

  This gives us a **Baseline** of roughly **88,65% accuracy**. **After Parameter Tuning**: **89,12%**




# Usage Instructions

1. Make sure to re-save the CSVs with UTF-8 encoding (e.g. with Sublime text on OSX), otherwise pandas cannot read it (see: https://stackoverflow.com/a/47922247).

2. (optional) For execution of the EDA notebook file, please start a jupyter notebook server on your local machine and execute each cell beginning from the top.

3. For generating the submission file from the *model.py* file, run `python model.py`

4. (optional) For execution of the Parameting Tuning notebook, please also start a jupyter notebook server on your local machine and, again, execute each cell beginning from the top (Note: Takes probably some hours)

### Dependencies:

Required (for Python 3.6.5):
  - xgboost (0.8.0)
  - numpy (1.15.1)
  - pandas (0.23.4)
  - scikit-learn (0.19.2)

Optional (additionally to the required dependencies):
   - matplotlib (2.2.3)
   - seaborn (0.9.0)
