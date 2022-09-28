# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 14:28:44 2020

@author: Franck
"""
#%% Import data

# Import usual modules
import pandas as pd
import numpy as np
import pingouin as pg
from scipy import stats
import statsmodels.api as sm

# plotting
import seaborn as sns 
import matplotlib.pyplot as plt

plt.style.use('seaborn')

pd.set_option('display.max_columns', 200) # affichage de plus de colonnes dans la console


# Import all the necessary CV and Performance modules
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance

#%% import rawdata in csv format

path = "F:/AP-HM/Rehospitalisation/Data"

data = pd.read_csv(path+"/sejour_sans_obstetrique&severitenondeterminee.csv", low_memory=False, index_col=0)

data.info()
data.head()

#%% Prepare the data for analysis

# Define the outcome variable
outcomeName =  "loscat"
y_data = data[outcomeName]
print(y_data.value_counts())
print(y_data.value_counts()/y_data.shape[0])

# Define the predictors
X_data = data.drop([outcomeName], axis=1)
X_data.info()

# #%%
# #==============================================
# # Create a SAS for choosing some variables
# #==============================================

# #%% Exclude Severity

# X_data = data.drop([outcomeName,"Severity"], axis=1)

# #%% Exclude Severity and Aggregated Charlson

# X_data = data.drop([outcomeName,"Severity","charlscore"], axis=1)

                    
# #%% Exclude Severity and the individual Charlson

# X_data = data.drop([outcomeName, "Severity",
#                     "CH_RENAL","CH_RD","CH_PVD","CH_PUD","CH_PLEGIA","CH_MSL",
#              "CH_MILD_LIVER_D","CH_METS","CH_MALIGNANCY","CH_HIV","CH_DM_COMP",
#              "CH_DM","CH_DEMENTIA","CH_CVD","CH_COPD","CH_CHF","CH_MI"], axis=1)


# #%%

# X_data.info()


#%% Generate onehot encoding

# Get dummies
X_dummy = pd.get_dummies(X_data, prefix_sep='_', drop_first=True)

# X head
X_dummy.info()

#%% Select only 20% of the full dataset for training

X_train_full, X_test, y_train_full, y_test = train_test_split(X_dummy, y_data, test_size=0.20, random_state=42, stratify=y_data)

# Check stratification frequency and proportion
print(y_train_full.value_counts())
print(y_train_full.value_counts()/y_train_full.shape[0])

#%%

#================================================
# Moving into modeling with different classifiers
#================================================

REPEAT = 10

#%%

# initialize elasticnet
lr_auc = []
lr_accuracy = []
lr_impMat = []
lr_proba = pd.DataFrame({'A' : []})
lr_class = pd.DataFrame({'A' : []})
lr_best = []
#lr_params = []

# Import the Logistic Regression Model
from sklearn.linear_model import LogisticRegression

for i in range(REPEAT) :
    print('\n''Iteration number :', i+1)
    # Create train and test sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=None, shuffle=True, stratify=y_train_full)


#--------------------
# Logistic Regression
#--------------------
   
    # Instantiate Logistic Regression
    best_lr = LogisticRegression(penalty='none', max_iter=200)

    # Fit Logistic Regression
    best_lr.fit(X_train, y_train)    
    
    # Select the best model

    lr_best.append(best_lr)
      
    # Compute predicted probabilities: y_pred_proba
    lr_pred_proba = pd.DataFrame(best_lr.predict_proba(X_val)[:,1])
    lr_pred_class = pd.DataFrame(best_lr.predict(X_val))
    lr_proba = pd.concat([lr_proba,lr_pred_proba], axis=1)
    lr_class = pd.concat([lr_class,lr_pred_class], axis=1)

    # Print the optimal parameters and best score
    # print("Regression Parameters: {}".format(best_lr.get_params()))
    print("Logistic Regression Accuracy: {}".format(accuracy_score(y_val, lr_pred_class)))
    print("Logistic Regression AUC: {}".format(roc_auc_score(y_val, lr_pred_proba)))

    # Save the parameters
    # lr_params.append(best_lr.get_params())
    
    # Compute scores
    lr_accuracy.append(accuracy_score(y_val, lr_pred_class))
    lr_auc.append(roc_auc_score(y_val, lr_pred_proba))
    
    lr_imp = permutation_importance(best_lr, X_val, y_val, scoring='roc_auc')
    lr_impMat.append(lr_imp.importances_mean)


#%%

#--------------------
# Elasticnet
#--------------------
# Compute mean probability prediction :

lr_proba_mean = lr_proba.mean(axis=1)
lr_class_mean = lr_class.mean(axis=1)
    
# Compute and print mean AUC score
lr_meanAccuracy = np.mean(lr_accuracy)
print("'\n' average Accuracy for the penalized Logistic Regression: {:.3f}".format(lr_meanAccuracy))

lr_meanAuc = np.mean(lr_auc) 
print("'\n' average AUC for the penalized Logistic Regression: {:.3f}".format(lr_meanAuc))

lr_scoreData = pd.DataFrame( {'Accuracy':[lr_meanAccuracy], 'ROC AUC':[lr_meanAuc]})
lr_impMean = np.mean(lr_impMat, axis=0)
lr_impData = pd.DataFrame({'Modalities':X_test.columns, 'Importance':lr_impMean/lr_impMean.max()*100})

lr_result = lr_impData.sort_values(by=['Importance'], ascending=False)[:20]

print("'\n' average Importance for the Logistic Regression '\n'{}".format(lr_result))

#%% Evaluating Test Sample
best_lr = lr_best[lr_auc.index(max(lr_auc))]
lr_test_pred = best_lr.predict(X_test)
lr_test_pred_proba = best_lr.predict_proba(X_test)
lr_test_accuracy = accuracy_score(y_test, lr_test_pred)
lr_test_auc = roc_auc_score(y_test, lr_test_pred_proba[:,1])

#lr_best_params = lr_params[lr_auc.index(max(lr_auc))]

print("'\n' Performance sur l'échantillon Test - Accuracy : {:.3f}".format(lr_test_accuracy))
print("'\n' Performance sur l'échantillon Test - ROC : {:.3f}".format(lr_test_auc))
#print("'\n' Best Parameters :", lr_best_params)

#%% Saving model
model_path = "F:/Dropbox/Travaux_JAOTOMBO/These_Sante_Publique/Projets/Duree_Sejour/Modeles"

from joblib import dump, load
dump(best_lr, model_path+"/logistic_regression.joblib")

#%% Test model importing

test_model = load(model_path+"/logistic_regression.joblib")

test_predict = test_model.predict(X_test)
test_predict_proba = test_model.predict_proba(X_test)
test_accuracy = accuracy_score(y_test, test_predict)
test_auc = roc_auc_score(y_test, test_predict_proba[:,1])

