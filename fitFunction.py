import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from dataProcess import load_data

# Get the dataset
x_train, x_test, y_train, y_test = load_data()

'''
Construct the fitness function
'''
def aucScore(x, model_name):
    xt = x_train   # train data
    yt = y_train
    xv = x_test  # test data
    yv = y_test

    # Define selected features
    xtrain = xt[:, x == 1]
    ytrain = yt  # Solve bug
    xvalid = xv[:, x == 1]
    yvalid = yv  # Solve bug
    weak_learners = [('randomForest', RandomForestClassifier(random_state=32)),
                     ('dt', DecisionTreeClassifier(random_state=32)),
                     ('graBoost', GradientBoostingClassifier(random_state=32)),
                     ('knn', KNeighborsClassifier()),
                     ('gn', GaussianNB()),
                     ('ligGBM', LGBMClassifier(random_state=32)),
                     ('CatBoost', CatBoostClassifier(random_state=32)),
                     ('AdaBoost', AdaBoostClassifier(random_state=32))]
    # Training
    if model_name == 'random_forest':
        mdl = RandomForestClassifier(random_state=32)
    elif model_name == 'dt':
        mdl = DecisionTreeClassifier(random_state=32)
    elif model_name == 'graBoost':
        mdl = GradientBoostingClassifier(random_state=32)
    elif model_name == 'knn':
        mdl = KNeighborsClassifier()
    elif model_name == 'gn':
        mdl = GaussianNB()
    elif model_name == 'ligGBM':
        mdl = LGBMClassifier(random_state=32)
    elif model_name == 'CatBoost':
        mdl = CatBoostClassifier(random_state=32)
    elif model_name == 'svm':
        mdl = svm.SVC(random_state=32)
    else:
        mdl = AdaBoostClassifier(random_state=32)
    mdl.fit(xtrain, ytrain)
    # Prediction
    ypred = mdl.predict(xvalid)
    auc = metrics.roc_auc_score(yvalid, ypred)
    # print('auc value is:', auc)
    return auc

# Select optimal features based on AUC
# AUC score & Feature size
def Fun(x):
    # Round x to 0 or 1
    x = np.round(x, 0).astype(int)
    # Number of selected features
    num_feat = np.sum(x == 1)
    # Solve if no feature selected
    if num_feat == 0:
        cost = 1E10
    else:
        # Get error rate
        aucValue = aucScore(x, 'knn')
        cost = 1/(aucValue+np.finfo(float).eps)
    return cost

# Select optimal features based on lowest error rate
# Error rate
def error_rate(x):
    # Parameters
    xt = x_train   # train data
    yt = y_train
    xv = x_test  # test data
    yv = y_test

    # Number of instances
    num_train = np.size(xt, 0)
    num_valid = np.size(xv, 0)
    # Define selected features
    xtrain = xt[:, x == 1]
    ytrain = yt  # Solve bug
    xvalid = xv[:, x == 1]
    yvalid = yv  # Solve bug
    # Training
    mdl = RandomForestClassifier(random_state=42)
    mdl.fit(xtrain, ytrain)
    # Prediction
    ypred = mdl.predict(xvalid)

    acc = metrics.accuracy_score(yvalid, ypred)
    error = 1 - acc
    return error

# Error rate & Feature size
def FunEr(x):
    x = np.round(x, 0).astype(int)
    # Parameters
    alpha = 0.99
    beta = 1 - alpha
    # Original feature size
    max_feat = len(x)
    # Number of selected features
    num_feat = np.sum(x == 1)
    # Solve if no feature selected
    if num_feat == 0:
        cost = 1
    else:
        # Get error rate
        error = error_rate(x)
        # Objective function
        cost = alpha * error + beta * (num_feat / max_feat)
    return cost

def aucCal(x):
    x = np.round(x, 0).astype(int)
    xt = x_train   # train data
    yt = y_train
    xv = x_test  # test data
    yv = y_test

    # Define selected features
    xtrain = xt[:, x == 1]
    ytrain = yt  # Solve bug
    xvalid = xv[:, x == 1]
    yvalid = yv  # Solve bug
    # Training
    mdl = RandomForestClassifier(random_state=42)
    mdl.fit(xtrain, ytrain)
    # Prediction
    ypred = mdl.predict(xvalid)
    auc = metrics.roc_auc_score(yvalid, ypred)
    # print('auc value is:', auc)
    return auc
