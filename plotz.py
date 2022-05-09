from operator import index
import numpy as np
import matplotlib.pyplot as plt
import pandas
import scipy.stats as stats
import pylab
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge

def hist_critical_temp(dframe, title):
    dframe['critical_temp'].plot(kind='hist', color = 'teal', bins = 10, alpha = 0.6)
    plt.axvline(dframe['critical_temp'].mean(), c='navy')
    plt.axvline(dframe['critical_temp'].median(),c='navy', linestyle='--')
    plt.axvline(dframe['critical_temp'].quantile(0.25),c='navy', linestyle=':')
    plt.axvline(dframe['critical_temp'].quantile(0.75),c='navy', linestyle=':')
    plt.title('Histogram ' + title)
    plt.xlabel('Critical temperature (K)')
    plt.show()

def draw_boxplot(lista, title, y_label):
    lista.boxplot()
    # plt.boxplot(list, vert=False, showmeans=True)
    plt.title(title)
    plt.ylabel(y_label)
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

    coord = pandas.DataFrame({'elem_prop': percentage, 'elem': sort_elem})
    coord.to_csv('elem_proportion.csv')


#plot correlation matrix
def corr_coeff(dframe):
    corr = dframe.corr()
    heatmap = sns.heatmap(dframe.corr()[['critical_temp']].sort_values(by='critical_temp', ascending=False),
            yticklabels=corr.columns.values,   
            vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title('Features Correlating with Critical Temperature')
    # matrix = sns.heatmap(corr, xticklabels=False, yticklabels=False, cmap='BrBG')
    fig = heatmap.get_figure()
    plt.show()


# Plot feature importance for PCA
def PCA_feat_imp(top_importances, top_features):
    plt.title('Feature Importances')
    plt.barh(range(len(top_importances)), top_importances, color='b', align='center')
    plt.yticks(range(len(top_importances)), top_features)
    plt.xlabel('Relative Importance')
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
def cumulative_variance_ratio(var_ratio, num_components):
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

# alpha parameters plot for ridge regression in semilog scale
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


# Three plots: 
# - Total Impurity vs effective alpha for training set; 
# - Number of nodes vs alpha;
# - Depth of tree vs alpha.
def alphas_path(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8)
    # Create Decision Tree Regressor object
    tree_reg = DecisionTreeRegressor(min_impurity_decrease=0.001)  #, ccp_alpha=0.618) 
    # Fit
    tree_reg.fit(X_train, y_train)
    path = tree_reg.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    plt.plot(ccp_alphas[:-2], impurities[:-2], marker="o", drawstyle="steps-post")
    # df = pd.DataFrame({'alpha': ccp_alphas[:-2], 'impurities': impurities[:-2]})
    # df.to_csv('alpha_vs_imp_noCu.csv')
    plt.xlabel("effective alpha")
    plt.ylabel("total impurity of leaves")
    plt.title("Total Impurity vs effective alpha for training set")
    plt.xscale('log')
    plt.show()
    ccp_alphas = np.abs(ccp_alphas)
    print(len(ccp_alphas))      
    
    clfs = []
    for ccp_alpha in ccp_alphas:
        tree_reg = DecisionTreeRegressor(ccp_alpha=ccp_alpha)
        tree_reg.fit(X_train, y_train)
        clfs.append(tree_reg)
        print(ccp_alpha)

    clfs = clfs[:-2]
    ccp_alphas = ccp_alphas[:-2]

    node_counts = [tree_reg.tree_.node_count for tree_reg in clfs]
    print(node_counts)
    depth = [tree_reg.tree_.max_depth for tree_reg in clfs]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
    # df2 = pd.DataFrame({'alpha': ccp_alphas, 'node': node_counts})
    # df2.to_csv('alpha_vs_node_noCu.csv')
    ax[0].set_xlabel("alpha")
    ax[0].set_xscale('log')
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
    # df3 = pd.DataFrame({'alpha': ccp_alphas, 'depth': depth})
    # df3.to_csv('alpha_vs_depth_noCu.csv')
    ax[1].set_xlabel("alpha")
    ax[1].set_xscale('log')
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs alpha")
    fig.tight_layout()
    plt.show()

# plot the chart of MSE versus number of estimators
def plt_mse_vs_nestim_tree(estimators, bagging_mse, rf_mse, bagging_mse2, rf_mse2, bagging_mse3, rf_mse3):
    plt.figure(figsize=(8, 8))
    plt.title('XGBoost for Cu and Not Cu')
    plt.plot(estimators, bagging_mse, 'b-', color="black", label='Bagging (Cu)')
    plt.plot(estimators, rf_mse, 'b-', color="blue", label='Random Forest (Cu)')
    plt.plot(estimators, bagging_mse2, 'b-', color="yellow", label='Bagging (Not Cu)')
    plt.plot(estimators, rf_mse2, 'b-', color="green", label='Random Forest (Not Cu)')
    plt.plot(estimators, bagging_mse3, 'b-', color="yellow", label='Bagging (All)')
    plt.plot(estimators, rf_mse3, 'b-', color="green", label='Random Forest (All)')

    # plt.plot(estimators, boosting_mse, 'b-', color="red", label='Boosting (Cu)')
    # plt.plot(estimators, boosting_mse2, 'b-', color="yellow", label='Boosting (Not Cu)')
    plt.legend(loc='upper right')
    plt.xlabel('Estimators')
    plt.ylabel('Mean Squared Error')
    plt.show()

# R2 train and R2 test comparison plot
def R2_train_test_plt(R2_train, R2_test, num_x, parameter):
    plt.figure(figsize=(8, 8))
    plt.title('R2 train and R2 test comparison')
    if parameter=='max_depth':
        plt.scatter(np.argmax(R2_test), np.max(R2_test), marker='*', color='red', zorder=2,
        label='Max R2 test: %.2f \nfor max_depth: %d' % (np.max(R2_test), np.argmax(R2_test))
        )
    plt.plot(num_x, R2_train, 'b-', color="#414487", label='R2 train', zorder=1)
    plt.plot(num_x, R2_test, 'b-', color="#FDE725", label='R2 test', zorder=1)
    plt.legend(loc='lower right')
    plt.xlabel(parameter)
    plt.ylabel('R2')
    plt.show()