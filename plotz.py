import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab
import seaborn as sns

def hist_critical_temp(dframe, title):
    dframe['critical_temp'].plot(kind='hist', color = 'teal', bins = 10, alpha = 0.6)
    plt.axvline(dframe['critical_temp'].mean(), c='navy')
    plt.axvline(dframe['critical_temp'].median(),c='navy', linestyle='--')
    plt.axvline(dframe['critical_temp'].quantile(0.25),c='navy', linestyle=':')
    plt.axvline(dframe['critical_temp'].quantile(0.75),c='navy', linestyle=':')
    plt.title('Histogram ' + title)
    plt.xlabel('Critical temperature (K)')
    plt.show()

def draw_boxplot(list, title, x_label):
    plt.boxplot(list, vert=False, showmeans=True)
    plt.title(title)
    plt.xlabel(x_label)
    plt.show()

def prob_plot(dframe, dist, name):
    if dist=='norm':
        stats.probplot(dframe['critical_temp'], dist=dist, plot=pylab)
        pylab.show()
    elif dist=='poisson':
        media = np.mean(dframe['critical_temp'])
        stats.probplot(dframe['critical_temp'], dist=dist, sparams=(media,), plot=pylab)
        pylab.show()

# plot of element proportion into the dataset
def plot_element_proportion(unique_m, Tc):
    elements = list(unique_m.columns)
    elements.remove('critical_temp')
    elements.remove('material')
    total_n = len(Tc)
    perc_elements = []
    for element in elements:
        n_element = len(unique_m[unique_m[element]!=0])
        perc = n_element/total_n
        perc_elements.append(perc)
    perc_dict = dict(zip(elements, perc_elements))
    sort_orders = sorted(perc_dict.items(), key=lambda x: x[1], reverse=True)
    count = 0
    coordinates = []
    sort_elem = []
    percentage = []
    for i in sort_orders:
        i = list(i)
        i.insert(0, count)
        sort_elem.append(i[1])
        percentage.append(i[2])
        coordinates.append(i)
        count = count + 1

    fig, ax = plt.subplots()
    ax.scatter(list(range(len(sort_elem))), percentage)

    for x in coordinates: plt.annotate(x[1], (x[0], x[2]))
    plt.xlabel(' ')
    plt.ylabel('Element proportion')
    plt.title('Proportions of the superconductors that had each element')
    plt.show()

#plot correlation matrix
def corr_matrix(dframe):
    corr = dframe.corr()
    matrix = sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
    fig = matrix.get_figure()
    plt.show()

# plot C_p, AIC, BIC and R_2_adj versus the number of features
def Plot_Cp_AIC_BIC_R2adj(dframe, dframe_name):
    variables = ['C_p', 'AIC','BIC','R_squared_adj']
    for v in variables:
        plt.plot(dframe['numb_features'],dframe[v], color = 'green')
        plt.scatter(dframe['numb_features'],dframe[v], color = 'orange')
        plt.ylabel(v)
        plt.xlabel('Number of predictors')
        plt.title('Subset selection using ' + v + ' from ' + dframe_name)
        plt.show()

# plot the cumulative variance versus number of components
def PCA_variance_ratio(var_ratio, num_components):
    cum_var = np.cumsum(var_ratio)
    plt.bar(range(1,num_components+1), var_ratio, alpha=0.5,
            align='center', label='individual explained variance')
    plt.step(range(1,num_components+1), cum_var, where='mid',
            label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.ylabel('Cumulative explained variance')
    plt.show()

# lasso coefficients as a function of alpha
from sklearn.linear_model import Lasso
def plot_lasso_alpha(X_train, y_train):
    alphas = np.linspace(0.01,100,100)
    lasso = Lasso(max_iter=1000)
    
    coefs = []
    for a in alphas:
        lasso.set_params(alpha=a)
        lasso.fit(X_train, y_train)
        coefs.append(lasso.coef_)
    
    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    plt.axis('tight')
    plt.xlabel('alpha')
    plt.ylabel('Standardized Coefficients')
    plt.title('Lasso coefficients as a function of alpha')
    plt.show()

from sklearn.linear_model import Ridge
def plot_ridge_alpha(x, y):
    n_alphas = 200
    alphas = np.logspace(-6, 6, n_alphas)

    coefs = []
    for a in alphas:
        ridge = Ridge(alpha=a, fit_intercept=False)
        ridge.fit(x, y)
        coefs.append(ridge.coef_[0])
    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale("log")
    plt.xlabel("alpha")
    plt.ylabel("weights")
    plt.title("Ridge coefficients as a function of the regularization")
    plt.axis("tight")
    plt.show()

# plot test MSE vs num of components 
def plot_mse_vs_ncomp(mse_PCA, mse_PLS):
    plt.plot(list(range(0,65)), mse_PCA, label='PC Regression')
    plt.plot(list(range(0,65)), mse_PLS, label='PLS Regression')
    plt.xlabel('Number of components')
    plt.ylabel('MSE')
    plt.legend(loc='best')
    plt.show()

# plot the chart of MSE versus number of estimators
def plt_mse_vs_nestim_tree(estimators, bagging_mse, rf_mse, boosting_mse):
    plt.figure(figsize=(8, 8))
    plt.title('Bagging, Random Forest and Boosting comparison')
    plt.plot(estimators, bagging_mse, 'b-', color="black", label='Bagging')
    plt.plot(estimators, rf_mse, 'b-', color="blue", label='Random Forest')
    plt.plot(estimators, boosting_mse, 'b-', color="red", label='Boosting')
    plt.legend(loc='upper right')
    plt.xlabel('Estimators')
    plt.ylabel('Mean Squared Error')
    plt.show()
