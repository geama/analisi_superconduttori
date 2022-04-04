
from tkinter import W
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import learning_curve, train_test_split, RepeatedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, scale
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text, export_graphviz
import graphviz
import scipy.stats as stats
import pylab
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor
import xgboost
from xgboost import plot_importance, plot_tree

from mlxtend.evaluate import bias_variance_decomp

# Standard decision tree
def decision_tree(x, y):
    '''
    This function performs Decision Tree Regressor after the tuning of alpha (fixed)
    and returns RMSE, R squared scores for training and test set and one ordered dictionary
    with features and importance score of features.
    '''
    # Splitto training set e test set 70/30
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8)

    # Create Decision Tree Regressor object
    tree_reg = DecisionTreeRegressor(ccp_alpha=0.618) 
    # Fit
    tree_reg.fit(X_train, y_train)
    # scores
    R2_train = tree_reg.score(X_train, y_train)
    R2_test = tree_reg.score(X_test, y_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, tree_reg.predict(X_train)))
    rmse_test = np.sqrt(mean_squared_error(y_test, tree_reg.predict(X_test)))
    # print('R2_train:', R2_train)
    # print('R2_test:', R2_test)
    # print('rmse_train:', rmse_train)
    # print('rmse_test:', rmse_test)
    # cross validation score
    cv_score = cross_val_score(tree_reg, x, y, cv=10)
    # tree rapresentation    
    f_name = list(tree_reg.feature_names_in_)
    text_representation = export_text(tree_reg, feature_names=f_name)
    # features importance
    f_imp = list(tree_reg.feature_importances_)
    feat_imp =  dict(zip(f_name, f_imp))
    featimp_ord = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
    # decision path
    dp = tree_reg.decision_path(x)
    # depth of tree
    depth = tree_reg.get_depth()
    n_leaves = tree_reg.get_n_leaves
    # print('Depth: ', depth)
    return R2_train, R2_test, rmse_train, rmse_test, featimp_ord
    

def cv_alpha(x,y):
    '''
    This function does the tuning of effective alpha using cost_complexity_pruning_path, that
    returns all alphas for each node of tree, then use this each alpha for learning of decision 
    tree regressor and calculate MSE and R squared scores for training and test sets and cross
    validation score (10-fold)
    '''
    # Splitto training set e test set 70/30
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7, shuffle=True)
    # Create Decision Tree Regressor object
    full_tree = DecisionTreeRegressor(min_impurity_decrease=0.01, random_state=0) 
    # Fit
    full_tree.fit(X_train, y_train)
    # prune the tree with cost complexity pruning — Alpha
    path = full_tree.cost_complexity_pruning_path(X_train, y_train)
    alphas, impurities = path.ccp_alphas, path.impurities
    alphas = np.abs(alphas)
    # mean, std = [], []
    R2_train_list = []
    R2_test_list = []
    rmse_test_list = []
    rmse_train_list = []
    mean = [] 
    count = 1
    for i in alphas:
        full_tree = DecisionTreeRegressor(ccp_alpha=i, min_impurity_decrease=0.01, random_state=0)
        full_tree.fit(X_train, y_train)
        print(i)
        R2_train = full_tree.score(X_train, y_train)
        R2_test = full_tree.score(X_test, y_test)
        rmse_train = np.sqrt(mean_squared_error(y_train, full_tree.predict(X_train)))
        rmse_test = np.sqrt(mean_squared_error(y_test, full_tree.predict(X_test)))
        R2_train_list.append(R2_train)
        R2_test_list.append(R2_test)
        rmse_test_list.append(rmse_test)
        rmse_train_list.append(rmse_train)

        # 10 fold cross validation for each alpha value
        scores = cross_val_score(full_tree, x, y, cv=10)
        mean.append(np.mean(scores))
            
    # keep a record of the values of alpha, mean accuracy rate, standard deviation of accuracies
    eva_df = pd.DataFrame({'alpha': alphas, 'mean cv': mean, 'R2_test': R2_test_list, 'R2_train': R2_train_list, 'RMSE_test': rmse_test_list, 'RMSE_train': rmse_train_list})
    eva_df = eva_df.sort_values(['alpha'], ascending = False)
    eva_df.to_csv('eva_df_not_Cu.csv')
    

def alpha_mse(x,y, reg):
    y = y.ravel()
    x = np.array(x)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)
    mse_train_list = []
    mse_test_list = []
    depth = []
    for i in list(np.arange(0, 4, 0.1)):
        regressor = reg(ccp_alpha=i)
        regressor.fit(X_train, y_train)
        mse_train = mean_squared_error(y_train, regressor.predict(X_train))
        mse_test = mean_squared_error(y_test, regressor.predict(X_test))
        mse_train_list.append(np.sqrt(mse_train))
        mse_test_list.append(np.sqrt(mse_test))
        depth.append(regressor.get_depth())
        print('Così', regressor.get_depth())

    # plt.plot(list(np.arange(0, 4, 0.1)), mse_train_list, label='RMSE train')
    # plt.plot(list(np.arange(0, 4, 0.1)), mse_test_list, label='RMSE test')
    # plt.xlabel('alpha')
    # plt.ylabel('RMSE')
    # plt.legend()
    # plt.show()
    
    # return mse_train_list, mse_test_list

def tuning_depth(x,y):
    '''
    This function does the tuning of tree depth for XGBoost
    fix tree depth from 2 to 20, increased by 1 for each  
    cycle using GridSearchCV + RepeatedKfold for calculation
    of cross validation score.
    '''
    # standardizzazione delle feature
    scalar = StandardScaler()
    x = scalar.fit_transform(x)
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)
    # define grid
    grid = dict()
    grid['max_depth'] = np.arange(2, 20, 1)
    # define search
    search = GridSearchCV(DecisionTreeRegressor, grid, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    # perform the search
    results = search.fit(x, y)
    return results.best_params_ 

def tuning_depth(x,y, model):
    '''
    This function does the tuning of tree depth for DecisonTree and XGBoost: 
    fix tree depth from 1 to 21, increased by 1 for each  
    cycle and record MSE and R square for test and train sets.
    '''
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)
    mse_train_list = []
    mse_test_list = []
    R2_train_list = []
    R2_test_list = []
    for i in range(1,21,1):
        if model=='xgb':
            xgb = xgboost.XGBRegressor(n_estimators=50, max_depth=i, learning_rate=0.22)
            xgb.fit(X_train, y_train)
            mse_train_list.append(mean_squared_error(y_train, xgb.predict(X_train)))
            mse_test_list.append(mean_squared_error(y_test, xgb.predict(X_test)))
            R2_train_list.append(xgb.score(X_train, y_train))
            R2_test_list.append(xgb.score(X_test, y_test))
        else:
            tree = DecisionTreeRegressor(max_depth=i)
            tree.fit(X_train, y_train)
            mse_train_list.append(mean_squared_error(y_train, tree.predict(X_train)))
            mse_test_list.append(mean_squared_error(y_test, tree.predict(X_test)))
            R2_train_list.append(tree.score(X_train, y_train))
            R2_test_list.append(tree.score(X_test, y_test))
    return mse_train_list, mse_test_list, R2_train_list, R2_test_list


def tuning_learning_rate(x,y):
    '''
    This function does the tuning of learning rate for XGBoost
    fix learning rate from 0.1 to 1.1, increased by 0.01 for each  
    cycle and record MSE, R squared and cross validation score (5-fold)
    '''
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)
    mse_train_list = []
    mse_test_list = []
    R2_train_list = []
    R2_test_list = []
    cv_score = []
    alpha = []
    for i in np.arange(0.1,1.1,0.01):
        xgb = xgboost.XGBRegressor(n_estimators=20, learning_rate=i)
        xgb.fit(X_train, y_train)
        mse_train_list.append(np.sqrt(mean_squared_error(y_train, xgb.predict(X_train))))
        mse_test_list.append(np.sqrt(mean_squared_error(y_test, xgb.predict(X_test))))
        R2_train_list.append(xgb.score(X_train, y_train))
        R2_test_list.append(xgb.score(X_test, y_test))
        cv_score.append(np.mean(cross_val_score(xgb, x, y, cv=5)))
        alpha.append(i)

    LR_tuning = pd.DataFrame({'alpha': alpha, 'mse_train': mse_train_list, 'mse_test': mse_test_list, 'R2_train': R2_train_list, 'R2_test': R2_test_list, 'CV_score': cv_score})
    LR_tuning.to_csv('LR_tuning.csv')
    return mse_train_list, mse_test_list, R2_train_list, R2_test_list 

def bagging_tree(x,y):
    '''
    This function performs Bagging ensemble and returns scores:
    R squared and RMSE for training and test sets and out of bag score.
    '''
    y = y.ravel()
    x = np.array(x)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)
    bag_reg = BaggingRegressor(n_estimators=20, oob_score=True, n_jobs=1)
    bag_reg.fit(X_train, y_train)
    y_pred = bag_reg.predict(X_test)
    R2_train = bag_reg.score(X_train, y_train)
    R2_test = bag_reg.score(X_test, y_test)
    mse_train = mean_squared_error(y_train, bag_reg.predict(X_train))
    mse_test = mean_squared_error(y_test, y_pred)
    return R2_train, R2_test, np.sqrt(mse_train), np.sqrt(mse_test), bag_reg.oob_score_

def random_forest(x,y):
    '''
    This function performs Random forest ensemble and returns scores:
    R squared and RMSE for training and test sets and out of bag score.
    '''
    y = y.ravel()
    x = np.array(x)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)
    random_fr = RandomForestRegressor(n_estimators=20, max_features=int(65/3), oob_score=True, n_jobs=1)
    random_fr.fit(X_train, y_train)
    R2_train = random_fr.score(X_train, y_train)
    R2_test = random_fr.score(X_test, y_test)
    mse_train = mean_squared_error(y_train, random_fr.predict(X_train))
    mse_test = mean_squared_error(y_test, random_fr.predict(X_test))

    return R2_train, R2_test, np.sqrt(mse_train), np.sqrt(mse_test), random_fr.oob_score_

def XGB(x,y):
    '''
    This function performs XGboost ensamble and returns scores:
    R squared and RMSE for training and test sets and plot the
    first seven features with them importance score.
    '''
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)
    xgb = xgboost.XGBRegressor(n_estimators=20, learning_rate=0.21)
    xgb.fit(X_train, y_train)
    R2_train = xgb.score(X_train, y_train)
    R2_test = xgb.score(X_test, y_test)
    mse_train = mean_squared_error(y_train, xgb.predict(X_train))
    mse_test = mean_squared_error(y_test, xgb.predict(X_test))
    # plot feature importance for the first seven features
    # plot_importance(xgb, max_num_features=7)
    # plt.show()
    return (R2_train, R2_test, np.sqrt(mse_train), np.sqrt(mse_test))

def bagforest(x,y,x2,y2, x3, y3):
    '''
    This function does the tuning of the number of estimators for Bagging and Random Forest
    ensemble, for each number calculate MSE and R squared on test set of all materials dataset,
    cuprate and cuprate-free dataset.
    '''
    n_estimators = 50
    y = y.ravel()
    y2 = y2.ravel()
    y3 = y3.ravel()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.3, train_size=0.7)  # 2
    X_train3, X_test3, y_train3, y_test3 = train_test_split(x3, y3, test_size=0.3, train_size=0.7)  # 3

    random_state = 42
    n_jobs = 1  # Parallelisation factor for bagging, random forests
    step_factor = 1
    axis_step = int(n_estimators/step_factor)
    # Pre-create the arrays which will contain the MSE for*
    # each particular ensemble method
    estimators = np.zeros(axis_step)
    # bagging scores
    bagging_mse = np.zeros(axis_step)
    bagging_r2 = np.zeros(axis_step)
    bagging_mse2 = np.zeros(axis_step)
    bagging_r22 = np.zeros(axis_step)
    bagging_mse3 = np.zeros(axis_step)
    bagging_r23 = np.zeros(axis_step)
    # random forest scores
    rf_mse = np.zeros(axis_step)
    rf_r2 = np.zeros(axis_step)
    rf_mse2 = np.zeros(axis_step)
    rf_r22 = np.zeros(axis_step)
    rf_mse3 = np.zeros(axis_step)
    rf_r23 = np.zeros(axis_step)

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
        R2 = r2_score(y_test, bagging.predict(X_test))

        bagging2 = BaggingRegressor(
            DecisionTreeRegressor(), 
            n_estimators=step_factor*(i+1),
            n_jobs=n_jobs,
            random_state=random_state
        )

        bagging2.fit(X_train2, y_train2) # 2
        mse2 = mean_squared_error(y_test2, bagging2.predict(X_test2)) # 2
        R22 = r2_score(y_test2, bagging2.predict(X_test2)) # 2

        bagging3 = BaggingRegressor(
            DecisionTreeRegressor(), 
            n_estimators=step_factor*(i+1),
            n_jobs=n_jobs,
            random_state=random_state
        )

        bagging3.fit(X_train3, y_train3) # 3
        mse3 = mean_squared_error(y_test3, bagging3.predict(X_test3)) # 3
        R23 = r2_score(y_test3, bagging3.predict(X_test3))

        estimators[i] = step_factor*(i+1)
        bagging_mse[i] = mse
        bagging_mse2[i] = mse2
        bagging_mse3[i] = mse3
        bagging_r2[i] = R2
        bagging_r22[i] = R22
        bagging_r23[i] = R23

    # Estimate the Random Forest MSE over the full number
    # of estimators, across a step size ("step_factor")
    for i in range(0, axis_step):
        print("Random Forest Estimator: %d of %d..." % (
            step_factor*(i+1), n_estimators)
        )
        rf = RandomForestRegressor(
            n_estimators=step_factor*(i+1),
            n_jobs=n_jobs,
            random_state=23, max_features=int(65/3))
        
        rf.fit(X_train, y_train) 
        mse = mean_squared_error(y_test, rf.predict(X_test))
        R2 = r2_score(y_test, rf.predict(X_test)) 


        rf2 = RandomForestRegressor(
            n_estimators=step_factor*(i+1),
            n_jobs=n_jobs,
            random_state=25, max_features=int(65/3)

        )

        rf2.fit(X_train2, y_train2) # 2
        mse2 = mean_squared_error(y_test2, rf2.predict(X_test2)) # 2
        R22 = r2_score(y_test2, rf2.predict(X_test2)) # 2

        rf3 = RandomForestRegressor(
            n_estimators=step_factor*(i+1),
            n_jobs=n_jobs,
            random_state=25, max_features=int(65/3)

        )

        rf3.fit(X_train3, y_train3) # 3
        mse3 = mean_squared_error(y_test3, rf3.predict(X_test3)) # 3
        R23 = r2_score(y_test3, rf3.predict(X_test3)) # 3


        estimators[i] = step_factor*(i+1)
        rf_mse[i] = mse
        rf_mse2[i] = mse2
        rf_mse3[i] = mse3
        rf_r2[i] = R2
        rf_r22[i] = R22
        rf_r23[i] = R23

    df = pd.DataFrame({'estimatori' : estimators, 'MSE Bagging Cu' : bagging_mse, 'MSE Random Forest Cu' : rf_mse \
        , 'MSE Bagging Not Cu' : bagging_mse2, 'MSE Random Forest Not Cu' : rf_mse2, 'MSE Bagging All' : bagging_mse3, 'MSE Random Forest All' : rf_mse3})
    df.to_csv('estimators_ensamble.csv')
    dr = pd.DataFrame({'estimatori' : estimators, 'R2 Bagging Cu' : bagging_r2, 'R2 Random Forest Cu' : rf_r2 \
        , 'R2 Bagging Not Cu' : bagging_r22, 'R2 Random Forest Not Cu' : rf_r22, 'R2 Bagging All' : bagging_r23, 'R2 Random Forest All' : rf_r23})
    dr.to_csv('estimators_ensamble_r2.csv')
    return estimators, bagging_mse, bagging_mse2, bagging_mse3, rf_mse, rf_mse2, rf_mse3

def booosting(x,y,x2,y2, x3, y3):
    '''
    This function does the tuning of the number of estimators for XGBoost
    for each number calculate MSE and R squared on test set of all materials dataset,
    cuprate and cuprate-free dataset.
    '''
    n_estimators = 50
    y = y.ravel()
    y2 = y2.ravel()
    y3 = y3.ravel()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.3, train_size=0.7)  # 2
    X_train3, X_test3, y_train3, y_test3 = train_test_split(x3, y3, test_size=0.3, train_size=0.7)  # 3

    random_state = 42
    n_jobs = 1  # Parallelisation factor for bagging, random forests
    step_factor = 1
    axis_step = int(n_estimators/step_factor)
    # Pre-create the arrays which will contain the MSE for*
    # each particular ensemble method
    estimators = np.zeros(axis_step)
    boosting_mse = np.zeros(axis_step)
    boosting_mse2 = np.zeros(axis_step)
    boosting_mse3 = np.zeros(axis_step)
    boosting_r2 = np.zeros(axis_step)
    boosting_r22 = np.zeros(axis_step)
    boosting_r23 = np.zeros(axis_step)
    # Estimate the Boosting tree MSE over the full number
    # of estimators, across a step size ("step_factor")
    for i in range(0, axis_step):
        print("Boosting Estimator: %d of %d..." % (
            step_factor*(i+1), n_estimators)
        )
        boosting = xgboost.XGBRegressor( 
            n_estimators=step_factor*(i+1), learning_rate=0.21,
            random_state=random_state
        )
        boosting2 = xgboost.XGBRegressor( 
            n_estimators=step_factor*(i+1), learning_rate=0.20,
            random_state=random_state
        )
        boosting3 = xgboost.XGBRegressor( 
            n_estimators=step_factor*(i+1), learning_rate=0.22,
            random_state=random_state
        )
        boosting.fit(X_train, y_train)
        mse = mean_squared_error(y_test, boosting.predict(X_test))
        r2 =r2_score(y_test, boosting.predict(X_test))

        boosting2.fit(X_train2, y_train2)
        mse2 = mean_squared_error(y_test2, boosting2.predict(X_test2))
        r22 =r2_score(y_test2, boosting2.predict(X_test2))

        boosting3.fit(X_train3, y_train3)
        mse3 = mean_squared_error(y_test3, boosting2.predict(X_test3))
        r23 =r2_score(y_test3, boosting3.predict(X_test3))

        estimators[i] = step_factor*(i+1)
        boosting_mse[i] = mse
        boosting_mse2[i] = mse2
        boosting_mse3[i] = mse3
        boosting_r2[i] = r2
        boosting_r22[i] = r22
        boosting_r23[i] = r23
        print(mse)
    df = pd.DataFrame({'estimatori' : estimators, 'MSE Boosting Cu' : boosting_mse, 'MSE Boosting Not Cu' : boosting_mse2, 'MSE Boosting All' : boosting_mse3})
    df.to_csv('estimators_mse_boosting.csv')
    dr = pd.DataFrame({'estimatori' : estimators, 'R2 Boosting Cu' : boosting_r2, 'R2 Boosting Not Cu' : boosting_r22,  'R2 Boosting All' : boosting_r23})
    dr.to_csv('estimators_r2_boosting.csv')

    return estimators, boosting_mse, boosting_mse2, boosting_mse3, boosting_r2, boosting_r22, boosting_r23