from cmath import sqrt
from tabnanny import verbose
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, scale
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text, export_graphviz
import graphviz
import scipy.stats as stats
import pylab
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor
import xgboost

# Standard decision tree
def decision_tree(x, y, m_d):
    # Splitto training set e test set 70/30
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)
    # Create Decision Tree Regressor object
    tree_reg = DecisionTreeRegressor(max_depth=m_d) 
    # Fit
    tree_reg.fit(X_train, y_train)
    # Predict
    y_pred = tree_reg.predict(X_test)
    # scores
    R2_train = tree_reg.score(X_train, y_train)
    R2_test = tree_reg.score(X_test, y_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, tree_reg.predict(X_train)))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    # features importance
    f_name = list(tree_reg.feature_names_in_)
    f_imp = list(tree_reg.feature_importances_)
    # tree rapresentation
    text_representation = export_text(tree_reg, feature_names=f_name)
    feat_imp =  dict(zip(f_name, f_imp))
    return R2_train, R2_test, rmse_train, rmse_test

def max_depth_mse(x,y, reg):
    y = y.ravel()
    x = np.array(x)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)
    rmse_train_list = []
    rmse_test_list = []
    for i in range(1, 35, 1):
        regressor = reg(max_depth=i)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        rmse_train = np.sqrt(mean_squared_error(y_train, regressor.predict(X_train)))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_train_list.append(rmse_train)
        rmse_test_list.append(rmse_test)
    return rmse_train_list, rmse_test_list

def tuning_depth(x,y):
    # standardizzazione delle feature
    scalar = StandardScaler()
    x = scalar.fit_transform(x)
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define grid
    grid = dict()
    grid['max_depth'] = np.arange(2, 20, 1)
    # define search
    search = GridSearchCV(DecisionTreeRegressor, grid, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    # perform the search
    results = search.fit(x, y)
    return results.best_params_ 
    
def bagging_tree(x,y):
    y = y.ravel()
    x = np.array(x)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)
    bag_reg = BaggingRegressor(n_estimators=200)
    bag_reg.fit(X_train, y_train)
    y_pred = bag_reg.predict(X_test)
    R2_train = bag_reg.score(X_train, y_train)
    R2_test = bag_reg.score(X_test, y_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, bag_reg.predict(X_train)))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    return R2_train, R2_test, rmse_train, rmse_test

def random_forest(x,y):
    y = y.ravel()
    x = np.array(x)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)
    random_fr = RandomForestRegressor(n_estimators=200, max_features=int(65/3))
    random_fr.fit(X_train, y_train)
    y_pred = random_fr.predict(X_test)
    R2_train = random_fr.score(X_train, y_train)
    R2_test = random_fr.score(X_test, y_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, random_fr.predict(X_train)))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))

    return R2_train, R2_test, rmse_train, rmse_test

def boost_reg(x,y):
    y = y.ravel()
    x = np.array(x)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)
    ada_fr = AdaBoostRegressor(n_estimators=150)
    ada_fr.fit(X_train, y_train)
    y_pred = ada_fr.predict(X_test)
    R2_train = ada_fr.score(X_train, y_train)
    R2_test = ada_fr.score(X_test, y_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, ada_fr.predict(X_train)))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))

    return R2_train, R2_test, rmse_train, rmse_test

from xgboost import plot_importance, plot_tree
def XGB(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)
    xgb = xgboost.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.1)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    R2_train = xgb.score(X_train, y_train)
    R2_test = xgb.score(X_test, y_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, xgb.predict(X_train)))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    #plot_importance(xgb)
    #plt.show()
    return (R2_train, R2_test, rmse_train, rmse_test)

def bagforestboost(x,y):
    n_estimators = 150
    x = scale(x)
    y = scale(y)
    y = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)
    random_state = 42
    n_jobs = 1  # Parallelisation factor for bagging, random forests
    step_factor = 1
    axis_step = int(n_estimators/step_factor)
    # Pre-create the arrays which will contain the MSE for
    # each particular ensemble method
    estimators = np.zeros(axis_step)
    bagging_mse = np.zeros(axis_step)
    rf_mse = np.zeros(axis_step)
    boosting_mse = np.zeros(axis_step)

    # Estimate the Bagging MSE over the full number
    # of estimators, across a step size ("step_factor")
    for i in range(0, axis_step):
        print("Bagging Estimator: %d of %d..." % (
            step_factor*(i+1), n_estimators)
        )
        bagging = BaggingRegressor(
            DecisionTreeRegressor(), 
            n_estimators=step_factor*(i+1),
            n_jobs=n_jobs,
            random_state=random_state
        )
        bagging.fit(X_train, y_train)
        mse = mean_squared_error(y_test, bagging.predict(X_test))
        estimators[i] = step_factor*(i+1)
        bagging_mse[i] = mse

    # Estimate the Random Forest MSE over the full number
    # of estimators, across a step size ("step_factor")
    for i in range(0, axis_step):
        print("Random Forest Estimator: %d of %d..." % (
            step_factor*(i+1), n_estimators)
        )
        rf = RandomForestRegressor(
            n_estimators=step_factor*(i+1),
            n_jobs=n_jobs,
            random_state=random_state
        )
        rf.fit(X_train, y_train)
        mse = mean_squared_error(y_test, rf.predict(X_test))
        estimators[i] = step_factor*(i+1)
        rf_mse[i] = mse

    # Estimate the Boosting tree MSE over the full number
    # of estimators, across a step size ("step_factor")
    for i in range(0, axis_step):
        print("Boosting Estimator: %d of %d..." % (
            step_factor*(i+1), n_estimators)
        )
        boosting = xgboost.XGBRegressor(
            n_estimators=step_factor*(i+1),
            random_state=random_state,
            learning_rate=0.01
        )
        boosting.fit(X_train, y_train)
        mse = mean_squared_error(y_test, boosting.predict(X_test))
        estimators[i] = step_factor*(i+1)
        boosting_mse[i] = mse
    return estimators, bagging_mse, rf_mse, boosting_mse