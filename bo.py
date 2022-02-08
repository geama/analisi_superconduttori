import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
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
    R_due = multi_reg.score(X_train, y_train)
    # Predizione sul test set
    y_pred_train = multi_reg.predict(X_train)
    y_pred_test = multi_reg.predict(X_test)
    # errore quadratico medio
    mse = mean_squared_error(y_test, y_pred_train)
    rss = mse * len(y)
    # Plot y_test vs y_pred
    #plt.plot(y_test, y_pred, 'bo')
    #plt.show()

    return rss, R_due

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

def alpha_tuning(x,y):
    # standardizzazione delle feature
    #scalar = StandardScaler()
    #x = scalar.fit_transform(x)
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define grid
    grid = dict()
    grid['alpha'] = np.arange(0, 1, 0.01)
    # define search
    search = GridSearchCV(Lasso(), grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # perform the search
    results = search.fit(x, y)
    return results.best_score_, results.best_params_

def lasso_reg(x,y):
    # standardizzazione delle feature
    scalar = StandardScaler()
    x = scalar.fit_transform(x)
    # Splitting training set and test set 70/30
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)
    # Create an instance of Lasso Regression implementation
    lasso = Lasso(alpha=0.01)
    # Fit
    lasso.fit(X_train, y_train)
    # Create the model score
    score_test, score_train = lasso.score(X_test, y_test), lasso.score(X_train, y_train)
    # Fit results
    intercetta = lasso.intercept_
    coeff = lasso.coef_
    return score_test, score_train, intercetta, coeff
    
# principal component analysis (PCA) for feature extraction
def principal_component_analysis(x,num_components):
    feat_name = list(x.columns)
    # standardizzazione delle feature
    scalar = StandardScaler()
    x = scalar.fit_transform(x)
    # PCA
    pca = PCA(n_components=num_components)
    x_pca = pca.fit_transform(x)
    x_pca = pca.transform(x)
    #components = pca.components_
    # ex_var_ratio is a vector of the variance explained by each dimension
    var_ratio = pca.explained_variance_ratio_

    loadings = pca.components_.T * np.sqrt(var_ratio)
    PC_list = []
    for i in range(1,num_components+1):
        PC_list.append('PC' + str(i))
    loading_matrix = pd.DataFrame(loadings, columns=PC_list, index=feat_name)

    # calculate covariance matrix
    #cov_mat = np.cov(x)
    # eigenvectors and eigenvalues of covariance matrix of features
    #eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

    #tot = sum(eigen_vals)
    #var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
    #cum_var_exp = np.cumsum(var_exp)
    
    return x_pca, var_ratio, loading_matrix
