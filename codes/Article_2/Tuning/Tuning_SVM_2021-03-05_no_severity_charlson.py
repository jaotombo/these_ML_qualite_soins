# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 13:39:58 2020

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

# # Define the predictors
# X_data = data.drop([outcomeName], axis=1)
# X_data.info()

# #%%
# #==============================================
# # Create a SAS for choosing some variables
# #==============================================

# #%% Exclude Severity

# X_data = data.drop([outcomeName,"Severity"], axis=1)

# #%% Exclude Severity and Aggregated Charlson

# X_data = data.drop([outcomeName,"Severity","charlscore"], axis=1)

                    
#%% Exclude Severity and the individual Charlson

X_data = data.drop([outcomeName, "Severity",
                    "CH_RENAL","CH_RD","CH_PVD","CH_PUD","CH_PLEGIA","CH_MSL",
              "CH_MILD_LIVER_D","CH_METS","CH_MALIGNANCY","CH_HIV","CH_DM_COMP",
              "CH_DM","CH_DEMENTIA","CH_CVD","CH_COPD","CH_CHF","CH_MI"], axis=1)


#%%

X_data.info()

#%% Generate onehot encoding

# Get dummies
X_dummy = pd.get_dummies(X_data, prefix_sep='_', drop_first=True)

# X head
X_dummy.info()

#%% Select only 20% of the full dataset for training

X_train_full, X_test, y_train_full, y_test = train_test_split(X_dummy, y_data, test_size=0.80, random_state=42, stratify=y_data)

# Check stratification frequency and proportion
print(y_train_full.value_counts())
print(y_train_full.value_counts()/y_train_full.shape[0])

#%%

#================================================
# Moving into modeling with different classifiers
#================================================

REPEAT = 10
#REPEAT = 1

#%%

# initialize SVM
svm_auc = []
svm_accuracy = []
svm_impMat = []
svm_proba = pd.DataFrame({'A' : []})
svm_class = pd.DataFrame({'A' : []})
svm_best = []
svm_params = []


for i in range(REPEAT) :
    print('\n''Iteration number :', i+1)
    # Create train and test sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=None, shuffle=True)


#------------------------
# Support Vector Machine
#------------------------

    # Import the SVM Classifier
    from sklearn.svm import SVC
    
    # Choose a sample space grid to optimize the model

    # param_svm = {'C':[50],
    #             'gamma': [0.35],
    #             'kernel':['rbf'],
    #             'probability': [True]
    #             }    
    
    param_svm = {'C':[0.1, 1, 10, 100],
                'gamma': [1, 0.1, 0.01, 0.001],
                'kernel':['rbf', 'poly', 'sigmoid'],
                'probability': [True]
                }
        
    # Instantiate the svm Classifier : svm
    svm = SVC()
    
    random_svm = RandomizedSearchCV(estimator=svm,
                        param_distributions=param_svm,
                        cv=10,
                        verbose=1,
                        n_jobs=14,
                        scoring=['accuracy','roc_auc'], refit = 'roc_auc')
    
    # Fit the model to the training data
    random_svm.fit(X_train, y_train)
    
    # Select the best model
    best_svm = random_svm.best_estimator_
    svm_best.append(best_svm)
    
    # Compute predisvmed probabilities: y_pred_proba
    svm_pred_proba = pd.DataFrame(best_svm.predict_proba(X_val)[:,1])
    svm_pred_class = pd.DataFrame(best_svm.predict(X_val))
    svm_proba = pd.concat([svm_proba,svm_pred_proba], axis=1)
    svm_class = pd.concat([svm_class,svm_pred_class], axis=1)
      
    # Save the parameters
    svm_params.append(random_svm.best_params_)
      
    # Compute scores
    svm_auc.append(roc_auc_score(y_val, svm_pred_proba))
    svm_accuracy.append(accuracy_score(y_val, svm_pred_class))

    # Print the optimal parameters and best score
    print("Tuned Gradient Boosting Classifier parameters: {}".format(random_svm.best_params_))
    print("Tuned Gradient Boosting Classifier Accuracy: {}".format(accuracy_score(y_val, svm_pred_class)))
    print("Tuned Gradient Boosting Classifier AUC: {}".format(roc_auc_score(y_val, svm_pred_proba)))
    
    svm_imp = permutation_importance(best_svm, X_val, y_val, scoring='roc_auc')
    svm_impMat.append(svm_imp.importances_mean)


#%%

#-----------------------
# Support Vector Machine
#-----------------------
# Compute mean probability prediction :

svm_proba_mean = svm_proba.mean(axis=1)
svm_class_mean = svm_class.mean(axis=1)
    
# Compute and print mean AUC score
svm_meanAccuracy = np.mean(svm_accuracy)
print("'\n' average Accuracy for the Support Vector Machine: {}".format(svm_meanAccuracy))

svm_meanAuc = np.mean(svm_auc) 
print("'\n' average AUC for the Support Vector Machine: {}".format(svm_meanAuc))

svm_scoreData = pd.DataFrame( {'Accuracy':[svm_meanAccuracy], 'ROC AUC':[svm_meanAuc]})
svm_impMean = np.mean(svm_impMat, axis=0)
svm_impData = pd.DataFrame({'Modalities':X_test.columns, 'Importance':svm_impMean/svm_impMean.max()*100})

svm_result = svm_impData.sort_values(by=['Importance'], ascending=False)[:20]

print("'\n' average Importance for the Support Vector Machine '\n'{}".format(svm_result))

#%% Evaluating Test Sample
best_svm = svm_best[svm_auc.index(max(svm_auc))]
svm_test_pred = best_svm.predict(X_test)
svm_test_pred_proba = best_svm.predict_proba(X_test)
svm_test_accuracy = accuracy_score(y_test, svm_test_pred)
svm_test_auc = roc_auc_score(y_test, svm_test_pred_proba[:,1])

svm_best_params = svm_params[svm_auc.index(max(svm_auc))]

print("'\n' Performance sur l'échantillon Test - Accuracy : {:.3f}".format(svm_test_accuracy))
print("'\n' Performance sur l'échantillon Test - ROC : {:.3f}".format(svm_test_auc))
print("'\n' Best Parameters :", svm_best_params)

#%% Saving model
model_path = "F:/Dropbox/Travaux_JAOTOMBO/These_Sante_Publique/Projets/Duree_Sejour/Modeles"

from joblib import dump, load
dump(best_svm, model_path+"/svm.joblib")

#%% Test model importing

test_model = load(model_path+"/svm.joblib")

test_predict = test_model.predict(X_test)
test_predict_proba = test_model.predict_proba(X_test)
test_accuracy = accuracy_score(y_test, test_predict)
test_auc = roc_auc_score(y_test, test_predict_proba[:,1])

