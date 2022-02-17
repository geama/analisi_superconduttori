import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import bo
import tree_bo
import plotz


unique_m = pd.read_csv('unique_m.csv')
train = pd.read_csv('train.csv')
Tc = unique_m['critical_temp']


# Descriptive analysis

# summary stats
#print(bo.stat_parameters(train, 'critical_temp'))

# Crea un dataframe con colonne: elemento predominante, numero di composti, Tc media
'''
elements = list(unique_m.columns)
elements.remove('critical_temp')
elements.remove('material')
matrix = list(unique_m[elements].max(axis=1))
elem = list(unique_m[elements].idxmax(axis=1))
max_elem_df = pd.DataFrame({'Elemento': elem, 'Concentrazione': matrix})
max_elem_df = pd.concat([max_elem_df, unique_m['critical_temp']], axis=1)
#max_elem_df.to_csv('max_elem_df.csv', index=False)
mean_temp = max_elem_df.groupby('Elemento')['critical_temp'].mean()
print(mean_temp)

len_comp = []
mean_Tc = []
for elem in elements:
    mdf = max_elem_df[max_elem_df['Elemento']==elem]
    len_mdf = int(len(mdf))
    mean_tc = mdf['critical_temp'].mean()
    len_comp.append(len_mdf)
    mean_Tc.append(mean_tc)
mean_tc_for_elem = pd.DataFrame({'Elemento predom': elements, 'Quanti composti': len_comp, 'Tc media': mean_Tc})
mean_tc_for_elem = mean_tc_for_elem.sort_values(by=['Quanti composti'])
mean_tc_for_elem.to_csv('mean_tc_for_elem.csv', index='False')
'''
# plotz.plot_element_proportion(unique_m, Tc)
# plotz.hist_critical_temp(train, 'all materials')
# plotz.prob_plot(train, 'norm', 'all materials')
# plotz.prob_plot(train, 'poisson', 'all materials')

magg_77 = train.query('critical_temp>77') #maggiore della temp critica dell'azoto liquido
min_77 = train.query('critical_temp<77') #minore della temp critica dell'azoto liquido
print(len(magg_77), len(min_77))

col = list(train.columns)
features = []
for i in col:
    if i!='critical_temp' and 'std' not in i:
        features.append(i)
print(len(features))

# Dataframe for elements with Fe
iron = unique_m[unique_m['Fe']!=0]
# Dataframe for elements without Fe
non_iron = unique_m[unique_m['Fe']==0]
# Dataframe for elements with Cu
cuprate = unique_m[unique_m['Cu']!=0]
# Dataframe for elements without Cu
non_cuprate = unique_m[unique_m['Cu']==0]

# summary stats
#print(bo.stat_parameters(non_cuprate, 'critical_temp'))

# Are the distributions normal?

# plotz.prob_plot(cuprate, 'norm', 'cuprate')
# plotz.prob_plot(non_cuprate, 'norm', 'non cuprate')

# Are the distribitions like Poisson distribution?

# plotz.prob_plot(cuprate, 'poisson', 'cuprate')
# plotz.prob_plot(non_cuprate, 'poisson', 'non cuprate')

# Draw histograms

# plotz.hist_critical_temp(cuprate, 'materials with Cu')
# plotz.hist_critical_temp(non_cuprate, 'materials without Cu')

# Draw boxplots
# Tc_class_elements = [cuprate['critical_temp'], non_cuprate['critical_temp']]
# plotz.draw_boxplot(Tc_class_elements, 'Materials with Cuprate (1) and without Cuprate (2)', 'Critical temp (K)')

Tc = np.array(Tc)
Tcp = Tc
Tc = Tc.reshape(-1,1)
y = Tc
x = train[[i for i in features]]

# correlation matrix
#plotz.corr_matrix(x)

# Predictive analysis
# benchmark: step zero consists the linear regression
#rss, r2_train = bo.multi_reg(x,y)
#R2_test, R2_train, mse_test, mse_train = bo.multi_reg(x,y)

# Features selection - backward stepwise

'''
df = bo.backward_stepwise(x, y)
df_min_RSS = df[df.groupby('numb_features')['RSS'].transform(min) == df['RSS']]
df_max_R2 = df[df.groupby('numb_features')['R_squared'].transform(max) == df['R_squared']]
df.to_csv('df.csv', index=False)
df_min_RSS.to_csv('df_min_RSS.csv', index=False)
df_max_R2.to_csv('df_max_R2.csv', index=False)

# calculate Cp, AIC, BIC, R2_adj for models chosen with best 
# R^2 and RSS, so adds the columns to the input dataframe
bo.calculate_Cp_AIC_BIC_R2_adj(df_max_R2, y, 65)
bo.calculate_Cp_AIC_BIC_R2_adj(df_min_RSS, y, 65)

# plot C_p, AIC, BIC and R_2_adj versus the number of 
# features for chosen models with max R2 or min RSS

plotz.Plot_Cp_AIC_BIC_R2adj(df_max_R2,'models with R^2 max')
plotz.Plot_Cp_AIC_BIC_R2adj(df_min_RSS,'models with RSS min')
'''
'''
# R summery of reg lin
y = Tcp
X2 = sm.add_constant(x)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())
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
#n_features_pca = bo.num_feat_tuning_pca(x,y)
#n_features_pca['n_components']

#x_pca, var_ratio, loading_matrix = bo.principal_component_analysis(x,n_features_pca['n_components'])
# Plot the cumulative variance versus number of components
#plotz.PCA_variance_ratio(var_ratio, n_features_pca['n_components'])

#Cumulative Variance explains for PCA
#pcr_var_rate=np.cumsum(np.round(var_ratio, decimals=4)*100)
#print(pcr_var_rate)

# ----------- PARTIAL LEAST SQUARE REGRESSION -----------
# Research of best number of features for PLS using cross validation
#n_features_pls = bo.num_feat_tuning_pca(x,y) 

#pls_var_rate, loading_matrix = bo.partial_least_square(x, y, n_features_pls['n_components'])

# Plot MSE of PLS and PCR versus number of components
'''
mse_PLS = bo.PLS_mse_for_nc(x,y)
mse_PCR = bo.PCR_mse_for_nc(x, y)
plotz.plot_mse_vs_ncomp(mse_PCR, mse_PLS)
'''

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

#print(tree_bo.decision_tree(x,y,8))
#print(tree_bo.bagging_tree(x,y))
#print(tree_bo.random_forest(x,y))
#print(tree_bo.XGB(x,y))

# confronto tra i tre modelli di tree
#estimators, bagging_mse, rf_mse, boosting_mse = tree_bo.bagforestboost(x,y)
#plotz.plt_mse_vs_nestim_tree(estimators, bagging_mse, rf_mse, boosting_mse)
