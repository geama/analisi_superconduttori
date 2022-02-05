import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
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
    y_pred = multi_reg.predict(X_test)
    # errore quadratico medio
    mse = mean_squared_error(y_test, y_pred)
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