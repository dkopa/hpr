"""
Created on Fri Oct 27 21:27:30 2017

@author: hasnat
# Using Alexandru Papiu - Regularized Linear Models
https://www.kaggle.com/apapiu/regularized-linear-models
I did not do anything till now ... I am just testing a submission.
Test-submission
"""

import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.stats.stats import pearsonr

##################################################################
###################### DATA PREPARATION ##########################
##################################################################
# Load training data
train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

## transform the skewed numeric features by taking log(feature + 1) - this will make the features more normal
#matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
#prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
#prices.hist()

#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

# Create Dummy variables for the categorical features
all_data = pd.get_dummies(all_data)

# Replace the numeric missing values (NaN's) with the mean of their respective columns
#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())

#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice
##################################################################
###################### MACHINE LEARNING ##########################
##################################################################
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

alphas = [1, 0.1, 0.001, 0.0005, 0.0001]
cv_lasso = [rmse_cv(Lasso(alpha=alpha)).mean() for alpha in alphas]
print('LASSOCV (RMSE): ' + str(format(min(cv_lasso), '.3f')))
# print(np.argmin(cv_lasso))
lasso_model = Lasso(alpha=alphas[np.argmin(cv_lasso)]).fit(X_train, y)
coef = pd.Series(lasso_model.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

# On test set : 
preds = np.expm1(lasso_model.predict(X_test))
solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})
solution.to_csv("hasnat_lasso_sol.csv", index = False)