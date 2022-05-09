import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import bo
import tree_bo
import plotz

# Reading of csv files
unique_m = pd.read_csv('unique_m.csv')
train = pd.read_csv('train.csv')
# Add a column 'material' -> column of all the chemical formulas
train['material'] = unique_m['material']

# Add a column of boolean values 1 (= material with Cu) and 0 (= material without Cu)
bo.labelling_Cu(train)
# Sub-dataframe with critical temperature and lables
# Tc = new_train['critical_temp']
Tc = train[['critical_temp', 'lab']]

# Delete duplicates
bo.delete_duplicate(train)

# create a new dataframe without duplicates
new_train = pd.read_csv('new_train.csv') #train without duplicate
bo.labelling_Cu(new_train)
# Creation of x and y
# list of features (p)
    # delete following columns: 
    # - 'critical_temp' because it is the y; 
    # - 'material' because it is a column of strings (chemical formulas); 
    # - 'duplicate' and 'lab' because they are just a flag;
    # - 'std' beacause standard deviations are not good features for prediction of Tc
bo.Raffaele(new_train)
col = list(train.columns)
features = []
for i in col:
    if i!='critical_temp' and  i!='material' and  i!='duplicate' and i!='lab' and 'std' not in i:
        features.append(i)
# print(len(features))

# create x and y for whole dataframe
Tc = train['critical_temp']
Tc = np.array(Tc)
Tcp = Tc
Tc = Tc.reshape(-1,1)
y = Tc
x = train[[i for i in features]]

# Check if there are null values
# bo.is_null(new_train)

# create x_cu and y_cu for cuprate dataframe
train_cu = new_train.query('lab==1')
train_cu.to_csv('train_cu.csv')
x_cu = train_cu[[i for i in features]]
Tc_cu = train_cu['critical_temp']
Tc_cu = np.array(Tc_cu)
y_cu = Tc_cu.reshape(-1,1)

# Create x_no_cu and y_no_cu for cuprate-free dataframe
train_no_cu = new_train.query('lab==0')
train_no_cu.to_csv('train_no_cu.csv')
x_no_cu = train_no_cu[[i for i in features]]
Tc_not_cu = train_no_cu['critical_temp']
Tc_not_cu = np.array(Tc_not_cu)
y_not_cu = Tc_not_cu.reshape(-1,1)

# Descriptive analysis

# summary stats
# print(bo.stat_parameters(train, 'critical_temp'))
# print(bo.stat_parameters(new_train, 'critical_temp'))

# Crea un dataframe con colonne: elemento predominante, numero di composti, Tc media, std
# bo.predom_elem(unique_m)

# plotz.plot_element_proportion(unique_m, Tc)
# plotz.hist_critical_temp(new_train, 'all materials')
# plotz.prob_plot(new_train, 'norm', 'all materials')

# # summary stats
# print(bo.stat_parameters(new_train.query('lab==1'), 'critical_temp'))
# print(bo.stat_parameters(new_train.query('lab==0'), 'critical_temp'))

# Are the distributions normal?
# plotz.prob_plot(new_train.query('lab==1'), 'norm', 'cuprate')
# plotz.prob_plot(new_train.query('lab==0'), 'norm', 'non cuprate')

# Draw histograms

# plotz.hist_critical_temp(new_train.query('lab==1'), 'materials with Cu')
# plotz.hist_critical_temp(new_train.query('lab==0'), 'materials hout Cu')

# cu = new_train.query('lab==1')
# no_cu =new_train.query('lab==0')
# print(bo.stat_parameters(cu, 'critical_temp'))
# print(bo.stat_parameters(no_cu, 'critical_temp'))

# Draw boxplots
# uno = pd.DataFrame(new_train['critical_temp'])
# uno = uno.rename(columns={"critical_temp": "All Elements"})
# due = pd.DataFrame(cu['critical_temp'])
# due = due.rename(columns={"critical_temp": "Cuprate"})
# tre = pd.DataFrame(no_cu['critical_temp'])
# tre = tre.rename(columns={"critical_temp": "Cuprate-free"})
# stella = pd.concat([uno, due, tre], axis=1)
# Tc_class_elements = pd.DataFrame(new_train['critical_temp'], cu['critical_temp'], no_cu['critical_temp'])
# print(type(Tc_class_elements))
# plotz.draw_boxplot(stella, 'Boxplot', 'Critical temp (K)')

# correlation coeff
# x.corr()[['critical_temp']].sort_values(by='critical_temp', ascending=False).to_csv('corr_coeff.csv')
# plotz.corr_coeff(new_train)

# Predictive analysis
# benchmark: step zero consists the linear regression
# rss, r2_train = bo.multi_reg(x,y)
# R2_test, R2_train, mse_test, mse_train = bo.multi_reg(x,y)

# Features selection - backward stepwise
'''
df = bo.backward_stepwise(x_no_cu, y_not_cu)
df_min_RSS = df[df.groupby('numb_features')['RSS'].transform(min) == df['RSS']]
df_max_R2 = df[df.groupby('numb_features')['R_squared'].transform(max) == df['R_squared']]
df.to_csv('df.csv', index=False)
df_min_RSS.to_csv('df_min_RSS.csv', index=False)
df_max_R2.to_csv('df_max_R2.csv', index=False)

# calculate Cp, AIC, BIC, R2_adj for models chosen with best 
# R^2 and RSS, so adds the columns to the input dataframe
bo.calculate_Cp_AIC_BIC_R2_adj(df_max_R2, y_not_cu, 65)
bo.calculate_Cp_AIC_BIC_R2_adj(df_min_RSS, y_not_cu, 65)

# plot C_p, AIC, BIC and R_2_adj versus the number of 
# features for chosen models with max R2 or min RSS
plotz.Plot_Cp_AIC_BIC_R2adj(df_max_R2,'models with R^2 max')
plotz.Plot_Cp_AIC_BIC_R2adj(df_min_RSS,'models with RSS min')
'''

# ----------- RIDGE REGRESSION -----------
# Research of best alpha for rigde regression using cross validation
#ridge_alpha = bo.alpha_tuning_ridge(x,y)

# plot ridge weight of features versus different alpha values
#plotz.plot_ridge_alpha(x,y)

# with the best alpha, we perform ridge regression
#print(bo.ridge_reg(x,y,ridge_alpha['alpha']))

# ----------- LASSO REGRESSION -----------
# Research of best alpha for lasso regression using cross validation
#lasso_alpha = bo.alpha_tuning_lasso(x,y)

# plot lasso weight of features versus different alpha values
#plotz.plot_lasso_alpha(x,y)

# with the best alpha, we perform lasso regression
#print(bo.lasso_reg(x,y,lasso_alpha['alpha']))

# ----------- PRINCIPAL COMPONENT REGRESSION -----------
# Research of best number of features for PCR using cross validation
# n_features_pca = bo.num_feat_tuning_pca(x,y)
# n_features_pca['n_components']

# PCA feature importance 
x_pca, var_ratio, loading_matrix, pca = bo.principal_component_analysis(x, 11)#n_features_pca['n_components'])
top_importances, top_features = bo.PCA_feat_imp(pca, x)
plotz.PCA_feat_imp(top_importances, top_features)

# # Plot the cumulative variance versus number of components
# plotz.cumulative_variance_ratio(var_ratio, n_features_pca['n_components'])

# # Cumulative Variance explains for PCA
# pcr_var_rate=np.cumsum(np.round(var_ratio, decimals=4)*100)
# print(pcr_var_rate)

# ----------- PARTIAL LEAST SQUARE REGRESSION -----------
# Research of best number of features for PLS using cross validation
# n_features_pls = bo.num_feat_tuning_pca(x_no_cu,y_not_cu) 

# var_ratio_pls, pls_var_rate, loading_matrix = bo.partial_least_square(x_cu, y_cu, 65)
# plotz.cumulative_variance_ratio(var_ratio_pls, 65)

# ----------- RIDGE VERSUS LASSO -----------
# Plot MSE of PLS and PCR versus number of components
# mse_PLS = bo.PLS_mse_for_nc(x,y)
# mse_PCR = bo.PCR_mse_for_nc(x, y)
# plotz.plot_mse_vs_ncomp(mse_PCR, mse_PLS)
# compute of loadings (the coefficients of the linear combination of the original variables 
# from which the principal components are constructed.) of PCA and PLS and create a csv file.
'''
loading_matrixPCA = pd.DataFrame()
loading_matrixPLS = pd.DataFrame()
for p in list(range(5,66,5)):
    x_pca, var_ratio, loading_matrixPCA_for_p = bo.principal_component_analysis(x,p)
    PCA_load = pd.concat([loading_matrixPCA, loading_matrixPCA_for_p], axis=0, ignore_index=False)
    x_score, y_score, loading_matrixPLS_for_p = bo.partial_least_square(x,y,p)
    PLS_load = pd.concat([loading_matrixPLS, loading_matrixPLS_for_p], axis=0, ignore_index=False)
PCA_load.to_csv('PCA_load.csv', index=False)
PLS_load.to_csv('PLS_load.csv', index=False)
'''

# Tuning max depth for basic decision tree
# mse_train_list, mse_test_list, R2_train_list, R2_test_list = tree_bo.tuning_depth(x,y, 'tree')

# plotz.MSE_train_test_plt(mse_train_list, mse_test_list, list(range(1,21,1)), 'max_depth')
# plotz.R2_train_test_plt(R2_train_list, R2_test_list, list(range(1,21,1)), 'max_depth')
# print(tree_bo.decision_tree(x,y))

# tree_bo.alpha_mse(x_cu,y_cu, DecisionTreeRegressor)
# print(tree_bo.bagging_tree(x_cu,y_cu))
# print(tree_bo.bagging_tree(x_no_cu,y_not_cu))
# feat_name = tree_bo.decision_tree(x_no_cu,y_not_cu)
# print(tree_bo.random_forest(x_no_cu,y_not_cu,feat_name))
# print(tree_bo.random_forest(x_no_cu,y_not_cu))

# Tuning max depth for xgboost
# mse_train_list, mse_test_list, R2_train_list, R2_test_list = tree_bo.tuning_depth(x,y, 'xgb')

# plotz.MSE_train_test_plt(mse_train_list, mse_test_list, list(range(1,21,1)), 'max_depth')
# plotz.R2_train_test_plt(R2_train_list, R2_test_list, list(range(1,21,1)), 'max_depth')
# print(tree_bo.XGB(x,y)) 

# Tuning learning rate
# mse_train_list, mse_test_list, R2_train_list, R2_test_list = tree_bo.tuning_learning_rate(x_no_cu,y_not_cu)
# plotz.MSE_train_test_plt(mse_train_list, mse_test_list, np.arange(0.1,1.1,0.01), 'learning rate')

# confronto tra i tre modelli di tree
# estimators, bagging_mse, bagging_mse2, bagging_mse3, rf_mse, rf_mse2, rf_mse3 = tree_bo.bagforest(x_cu,y_cu, x_no_cu, y_not_cu, x, y)
# tree_bo.bagforest(x_cu,y_cu, x, y, x_no_cu, y_not_cu)
# plotz.plt_mse_vs_nestim_tree(estimators, bagging_mse, bagging_mse2, bagging_mse3, rf_mse, rf_mse2, rf_mse3)

# estimators, boosting_mse, boosting_mse2, boosting_mse3, boosting_r2, boosting_r22, boosting_r23 = tree_bo.booosting(x_cu,y_cu, x_no_cu, y_not_cu, x, y)

# effective alphas for whole tree
# tree_bo.cv_alpha(x_no_cu, y_not_cu)
# tree = tree_bo.decision_tree(x,y)

# tree_bo.cv_alpha(x,y)

# print(tree_bo.XGB(x_cu,y_cu))