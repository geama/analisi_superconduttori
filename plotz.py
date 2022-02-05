import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab

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
    elements = ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn',	'Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn']
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