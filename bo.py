import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, scale
import scipy.stats as stats
import pylab
import statsmodels.api as sm

def stat_parameters(dframe, feature):
    size = len(dframe[feature])
    max = np.max(dframe[feature])
    min = np.min(dframe[feature])
    std = np.std(dframe[feature])
    media = np.mean(dframe[feature])
    quartile_25 = np.quantile(dframe[feature], 0.25)
    median = np.quantile(dframe[feature], 0.50)
    quartile_75 = np.quantile(dframe[feature], 0.75)
    stats.kurtosis(dframe[feature])
    # Fisher-Pearson coefficient of skewness
    stats.skew(dframe[feature])
    return (size, min, quartile_25, median, quartile_75, max, media, std)

def multi_reg(x, y):
    # Splitto training set e test set 70/30
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)

    # Multi linear regression
    multi_reg = LinearRegression()

    # Regressione sul training set 
    multi_reg.fit(X_train, y_train)
    coeff = multi_reg.coef_
    intercetta = multi_reg.intercept_

    # R^2
    R2_train = multi_reg.score(X_train, y_train)
    R2_test = multi_reg.score(X_train, y_train)

    # Predizione sul test set
    y_pred_train = multi_reg.predict(X_train)
    y_pred_test = multi_reg.predict(X_test)
    # errore quadratico medio
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    rss = mse_train * len(y)
    # Plot y_test vs y_pred
    #plt.plot(y_test, y_pred, 'bo')
    #plt.show()
    #return mse_train, R2_train, mse_test, R2_test
    return rss, R2_train

# Forward stepwise for the models selection (you can use RSS 
# or R^2 to choose the variable to remove at each iteration,
# this function use min RSS.

def backward_stepwise(x, y):
    result_df = pd.DataFrame()
    for i in range(len(x.columns)-1):
        numb_features = len(x.columns)
        col = list(x.columns)
        all_features = []
        for feat in col:
            xp = x.drop(feat, axis=1)
            rss, r2 = multi_reg(xp, y)
            feat_name = ', '.join(list(x.columns))
            all_features.append([feat, rss, r2, feat_name])
            xp = x
        # adds the results in a dataframe
        feature_list, RSS_list, R_squared_list, = [i[3] for i in all_features], [i[1] for i in all_features], [i[2] for i in all_features]
        result_df_for_k = pd.DataFrame({'numb_features': numb_features,'RSS': RSS_list, 'R_squared':R_squared_list,'features':feature_list})
        result_df = pd.concat([result_df, result_df_for_k], axis=0, ignore_index=False)
        # you can use R^2 max or RSS min for choose the feature to delete
        #best_R2 = np.max(R_squared_list)
        #index_R2 = R_squared_list.index(best_R2)
        best_rss = np.min(RSS_list)
        index_rss = RSS_list.index(best_rss)
        del_feat = all_features[index_rss][0]
        x = x.drop(del_feat, axis=1)
    return result_df

# Calcolo C_p, AIC, BIC and R_2_adj
def calculate_Cp_AIC_BIC_R2_adj(dframe, y, num_of_features):
    m = len(y)
    p = num_of_features
    hat_sigma_squared = (1/(m - p -1)) * min(dframe['RSS'])

    dframe['C_p'] = (1/m) * (dframe['RSS'] + 2 * dframe['numb_features'] * hat_sigma_squared )
    dframe['AIC'] = (1/(m*hat_sigma_squared)) * (dframe['RSS'] + 2 * dframe['numb_features'] * hat_sigma_squared )
    dframe['BIC'] = (1/(m*hat_sigma_squared)) * (dframe['RSS'] +  np.log(m) * dframe['numb_features'] * hat_sigma_squared )
    dframe['R_squared_adj'] = 1 - ( (1 - dframe['R_squared'])*(m-1)/(m-dframe['numb_features'] -1))
    #print(df['R_squared_adj'].max())

def alpha_tuning_lasso(x,y):
    # standardizzazione delle feature
    scalar = StandardScaler()
    x = scalar.fit_transform(x)
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define grid
    grid = dict()
    grid['alpha'] = np.arange(0, 1, 0.01)
    # define search
    search = GridSearchCV(Lasso(), grid, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    # perform the search
    results = search.fit(x, y)
    return results.best_params_

def lasso_reg(x,y, alpha):
    # standardizzazione delle feature
    scalar = StandardScaler()
    x = scalar.fit_transform(x)
    # Splitting training set and test set 70/30
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)
    # Create an instance of Lasso Regression implementation
    lasso = Lasso(alpha=alpha)
    # Fit
    lasso.fit(X_train, y_train)
    y_pred_train = lasso.predict(X_train)
    y_pred_test = lasso.predict(X_test)
    # Create the model score
    R2_test, R2_train = lasso.score(X_test, y_test), lasso.score(X_train, y_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    # Fit results
    intercetta = lasso.intercept_
    coeff = lasso.coef_
    return R2_test, R2_train, mse_test, mse_train

def alpha_tuning_ridge(x,y):
    # standardizzazione delle feature
    scalar = StandardScaler()
    x = scalar.fit_transform(x)
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define grid
    grid = dict()
    grid['alpha'] = np.arange(0.0001, 100, 10)
    # define search
    search = GridSearchCV(Ridge(), grid, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    # perform the search
    results = search.fit(x, y)
    return results.best_params_

def ridge_reg(x,y, alpha):
    # standardizzazione delle feature
    scalar = StandardScaler()
    x = scalar.fit_transform(x)
    # Splitting training set and test set 70/30
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)
    # Create an instance of Ridge Regression implementation
    ridge = Ridge(alpha=alpha)
    # Fit
    ridge.fit(X_train, y_train)
    y_pred_train = ridge.predict(X_train)
    y_pred_test = ridge.predict(X_test)
    # Create the model score
    R2_test, R2_train = ridge.score(X_test, y_test), ridge.score(X_train, y_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    # Fit results
    intercetta = ridge.intercept_
    coeff = ridge.coef_
    return R2_test, R2_train, mse_test, mse_train

def num_feat_tuning_pca(x,y):
    # standardizzazione delle feature
    scalar = StandardScaler()
    x = scalar.fit_transform(x)
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define grid
    grid = dict()
    grid['n_components'] = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
    # define search
    search = GridSearchCV(PCA(), grid, cv=cv, n_jobs=-1)
    # perform the search
    results = search.fit(x,y)
    return results.best_params_

# principal component analysis (PCA) for feature extraction
def principal_component_analysis(x,num_components):
    feat_name = list(x.columns)
    # standardizzazione delle feature
    scalar = StandardScaler()
    x = scalar.fit_transform(x)
    # PCA*
    pca = PCA(n_components=num_components)
    x_pca = pca.fit_transform(x)
    #components = pca.components_
    # ex_var_ratio is a vector of the variance explained by each dimension
    var_ratio = pca.explained_variance_ratio_

    loadings = pca.components_.T * np.sqrt(var_ratio)
    PC_list = []
    for i in range(1,num_components+1):
        PC_list.append('PC' + str(i))
    loading_matrix = pd.DataFrame(loadings, columns=PC_list, index=feat_name)

    return x_pca, var_ratio, loading_matrix

def num_feat_tuning_pls(x,y):
    # standardizzazione delle feature
    scalar = StandardScaler()
    x = scalar.fit_transform(x)
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define grid
    grid = dict()
    grid['n_components'] = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
    # define search
    search = GridSearchCV(PLSRegression(), grid, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    # perform the search
    results = search.fit(x,y)
    return results.best_params_

def partial_least_square(x,y,num_components):
    # standardizzazione delle feature
    feat_name = list(x.columns)
    scalar = StandardScaler()
    x = scalar.fit_transform(x)
    # PLS
    pls = PLSRegression(n_components=num_components)
    x_score, y_score = pls.fit_transform(x,y)
    x_load = pls.x_loadings_

    # Splitting training set and test set 70/30
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)
    pls.fit(X_train, y_train)
    y_pred = pls.predict(X_test)

    PC_list = []
    for i in range(1,num_components+1):
        PC_list.append('PLS' + str(i))
    loading_matrix = pd.DataFrame(x_load, columns=PC_list, index=feat_name)

    # Variance explained by each component 
    r2_sum = 0
    r2_list = []
    for i in range(0,num_components):
        Y_pred=np.dot(pls.x_scores_[:,i].reshape(-1,1),pls.y_loadings_[:,i].reshape(-1,1).T)*y_train.std(axis=0, ddof=1)+y_train.mean(axis=0)
        r2_sum += round(r2_score(y_train,Y_pred),5) 
        r2_for_comp = round(r2_score(y_train,Y_pred),5) 
        r2_list.append(r2_for_comp)
    # Variance ratio
    var_ratio = []
    for i in r2_list:
        var_ratio.append(i/r2_sum)
    # Cumulative Variance explains
    pls_var_rate=np.cumsum(np.round(var_ratio, decimals=4)*100)
    return pls_var_rate, loading_matrix

def PCR_mse_for_nc(x,y):
    #scale predictor variables
    pca = PCA()
    regr = LinearRegression()
    X_reduced = pca.fit_transform(scale(x))
    #define cross validation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    mse = []
    # Calculate MSE with only the intercept
    score = -1*cross_val_score(regr,
            np.ones((len(X_reduced),1)), y, cv=cv,
            scoring='neg_mean_squared_error').mean()    
    mse.append(score)

    # Calculate MSE using cross-validation, adding one component at a time
    for i in np.arange(1, 65):
        score = -1*cross_val_score(regr,
                X_reduced[:,:i], y, cv=cv, scoring='neg_mean_squared_error').mean()
        mse.append(score)
    
    return mse

def PLS_mse_for_nc(x,y):
    #define cross-validation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    mse = []
    n = len(x)

    # Calculate MSE with only the intercept
    score = -1*cross_val_score(PLSRegression(n_components=1),
            np.ones((n,1)), y, cv=cv, scoring='neg_mean_squared_error').mean()    
    mse.append(score)

    # Calculate MSE using cross-validation, adding one component at a time
    for i in np.arange(1, 65):
        pls = PLSRegression(n_components=i)
        score = -1*cross_val_score(pls, scale(x), y, cv=cv,
                scoring='neg_mean_squared_error').mean()
        mse.append(score)

    return mse
