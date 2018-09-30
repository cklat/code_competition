#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV 

import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')




#load data
def load_csv():
    train_df = pd.read_csv("../data/verkehrsunfaelle_train.csv").drop(['Unnamed: 0'],axis=1)
    test_df = pd.read_csv("../data/verkehrsunfaelle_test.csv").drop(['Unnamed: 0'],axis=1)
    
    return train_df, test_df



# collective Function for renaming Columns and Feature values of the datasetto ensure naming consistency
def rename_data(df):
    renamed_df = df.copy()
    
    #Rename Time (="Zeit") column
    renamed_df.rename(columns={"Zeit (24h)": "Zeit"}, inplace=True)
    
    #Correcting notation of values from the Feature 'Bodenbeschaffenheit'
    renamed_df["Bodenbeschaffenheit"].replace("Frost/ Ice", "Frost / Eis", inplace=True)
    
    return renamed_df
    



# function to collectively filter rows that seem to be unimportant according to the EDA notebook
def clear_rows(df):
    cleared_df = df.copy()
    
    #drop the instance with the Feature value "Bodenbeschaffenheit = 9"
    cleared_df.drop(cleared_df.loc[cleared_df["Bodenbeschaffenheit"]=="9"].index, inplace=True)
    
    #drop instances with Feature value "Fahrzeugtyp = Unbekannt/Pferd/Traktor"
    cleared_df.drop(cleared_df.loc[cleared_df["Fahrzeugtyp"]=="Unbekannt"].index, inplace=True)
    cleared_df.drop(cleared_df.loc[cleared_df["Fahrzeugtyp"]=="Pferd"].index, inplace=True)
    cleared_df.drop(cleared_df.loc[cleared_df["Fahrzeugtyp"]=="Traktor"].index, inplace=True)

    #drop instances with Feature value "Wetterlage = Schnee (starker Wind)"
    cleared_df.drop(cleared_df.loc[cleared_df["Wetterlage"]=="Schnee (starker Wind)"].index, inplace=True)
                                                  
    return cleared_df




# Function to process and convert the date representations in the dataset and perform one-hot-encoding
def process_date(df, use_day=False):
    dateconverted_df = df.copy()
    dateconverted_df["Unfalldatum"] = dateconverted_df["Unfalldatum"].apply(lambda x: x[:x.rfind('-')] + '-2016')                                                  .apply(lambda x: x.replace(". ", "-"))                                                  .apply(lambda x: x.split('.', 1)[0])[:]


    conversions = {"Mrz": "Mar",
                   "Mai": "May",
                   "Okt": "Oct",
                   "Dez": "Dec"}

    dateconverted_df["Unfalldatum"] = dateconverted_df["Unfalldatum"].replace(conversions, regex=True)
    dateconverted_df["Unfalldatum"] = pd.to_datetime(dateconverted_df["Unfalldatum"], dayfirst=True)
    
    #one-hot-encode month values of date Feature
    dateconverted_df["Monat"] = dateconverted_df["Unfalldatum"].dt.month

    dateconverted_df = pd.get_dummies(dateconverted_df, columns=["Monat"])
    
    if use_day == True:
        dateconverted_df["Tag"] = dateconverted_df["Unfalldatum"].dt.day
        dateconverted_df = pd.get_dummies(dateconverted_df["Unfalldatum"].dt.day)
 
    dateconverted_df.drop('Unfalldatum', axis=1, inplace=True)
    
    return dateconverted_df



#Process the time feature and perform one-hot-encoding to make it usable for the algorithm
def process_time(df):
    timeconverted_df = df.copy()

    def append_zeros(x):
        if len(str(x)) == 3:
            return "0" + str(x)[0] + ":" + str(x)[-2:]
        if len(str(x)) == 2:
            return "00:" + str(x)
        if len(str(x)) == 4:
            return str(x)[:2] + ":" + str(x)[-2:]
    
    timeconverted_df["Zeit"] = timeconverted_df["Zeit"].apply(append_zeros)
    timeconverted_df["Zeit"] = pd.to_datetime(timeconverted_df["Zeit"], format="%H:%M")
    
    #We drop the minutes of the time representations in order to able to perform a one-hot-encoding. Keeping
    #the minutes would make the amount of unique values for this feature very large 
    #and we would generate a great number of new one-hot-encoded columns. 
    #It's fair to assume that the exact minute of an accident shouldn't be a reasonable predictor
    timeconverted_df["Stunde"] = timeconverted_df["Zeit"].dt.hour
    timeconverted_df = pd.get_dummies(timeconverted_df, columns=["Stunde"])
    
    timeconverted_df.drop("Zeit", axis=1, inplace=True)
    
    return timeconverted_df

                                                         



ONE_HOT_COLS = ["Strassenklasse", "Unfallklasse", "Lichtverhältnisse", "Bodenbeschaffenheit", "Geschlecht", 
               "Fahrzeugtyp", "Wetterlage"]

#Collective one-hot-enconding function for categorical columns
def one_hot_encoder(df):
    orig_cols = list(df.columns)
    df = pd.get_dummies(df, columns=ONE_HOT_COLS)

    return df 


#Function to train a model on the training dataset
def train_model(model, train_df):
    
    features = [f for f in train_df.columns if f not in ["Unfallschwere"]]
    
    model.fit(dtrain[features], dtrain["Unfallschwere"])
    
    return model


#Function to generate the submission file
def generate_submission(model, test_df):
    
    #Predict test set:
    features = [f for f in test_df.columns]

    test_predictions = model.predict(test_df[features])
    #test_predprob = alg.predict_proba(test_df[features])[:,1]
    
    test_predictions = pd.DataFrame(test_predictions).reset_index()
    
    test_predictions.to_csv(path_or_buf="../submission.csv", header=["Unfall_ID", "Unfallschwere"], index=False)


#Function to prepare the datasets with the necessary pre-processing steps. 
def prepare_df(train_df, test_df):
    
    train_df = clear_rows(train_df)
    
    df = train_df.append(test_df).reset_index()
    df = rename_data(df)
    df = one_hot_encoder(df)
    df = process_time(df)
    df = process_date(df)
    
    train_df = df[0:-1000]
    test_df = df[-1000:df.shape[0]]
    train_df = train_df.drop(['index'],axis=1)
    test_df = test_df.drop(['Unfallschwere'],axis=1)
    test_df = test_df.drop(['index'],axis=1)
    
    return train_df, test_df
    

train_df, test_df = load_csv()
dtrain, dtest = prepare_df(train_df, test_df)

xgb = XGBClassifier(
 learning_rate =0.1,
 n_estimators=100,
 num_class=4,
 max_depth=4,
 min_child_weight=2,
 gamma=0,
 subsample=0.7,
 colsample_bytree=0.6,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 reg_alpha=0.2,
 reg_lambda=1e-05,
 seed=1337)

model = train_model(xgb, dtrain)

generate_submission(model, dtest)




